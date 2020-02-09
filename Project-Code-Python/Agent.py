from collections import OrderedDict

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


class Agent:

    def __init__(self):
        self.weights = {"Unchanged": 6,
                        "Inverted": 5,
                        "Mirrored": 4,
                        "Rotated": 3,
                        "Scaled": 2,
                        "Deleted": 1,
                        "Changed": 0,
                        }

        self.tests = OrderedDict([
            ("Unchanged", [self.checkIdentical]),
            ("Inverted", [self.checkInverted]),
            ("Mirrored",
             [self.checkMirrored('horizontal'),
              self.checkMirrored('vertical')]),
            ("Rotated", [self.checkRotated(90),
                         self.checkRotated(180),
                         self.checkRotated(270)]),
            ("Scaled", [self.checkScaled]),
            ("Deleted", [self.checkDeleted]),
            ("Changed", [self.checkChanged])
        ])

    def getImages(self, problem):
        figures = problem.figures.items()
        probImgs = []
        ansImgs = []

        for figure in figures:
            key = figure[0]
            fileName = figure[1].visualFilename
            img = Image.open(fileName)
            if key.isalpha():
                probImgs.append((key, img))
            else:
                ansImgs.append((key, img))
        probImgs.sort(key=lambda entry: entry[0])
        ansImgs.sort(key=lambda entry: entry[0])
        return probImgs, ansImgs

    def checkCorrelation(self, img1, img2):
        hist1 = img1.histogram()
        hist2 = img2.histogram()
        minima = np.minimum(hist1, hist2)
        return np.true_divide(np.sum(minima), np.sum(hist2))

    def checkIdentical(self, img1, img2):
        img1 = img1.convert("1")
        img2 = img2.convert("1")
        if img1.size != img2.size:
            return False
        count = 0
        rows, cols = img1.size
        for row in range(rows):
            for col in range(cols):
                img1_pixel = img1.getpixel((row, col))
                img2_pixel = img2.getpixel((row, col))
                if img1_pixel != img2_pixel:
                    count += 1
                    if count > (rows * cols * 0.03):
                        return False
        return True

    def checkIdenticalBBox(self, img1, img2):
        inverted1 = ImageChops.invert(img1)
        inverted2 = ImageChops.invert(img2)
        box1 = img1.convert('RGB').crop(inverted1.getbbox())
        box2 = img2.convert('RGB').crop(inverted2.getbbox())
        if box1.size != box2.size:
            return False
        count = 0
        rows, cols = box1.size
        for row in range(rows):
            for col in range(cols):
                box1_pixel = box1.getpixel((row, col))
                box2_pixel = box2.getpixel((row, col))
                if box1_pixel != box2_pixel:
                    count += 1
                    if count > (rows * cols * 0.03):
                        print('pixel count: ' + str(count)
                              + 'limit: ' + str(rows * cols * 0.04))
                        return False
        return True

    def checkInverted(self, img1, img2):
        if self.checkCorrelation(img1, img2) > 0.96:
            return False

        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        inverted1 = ImageChops.invert(img1)
        inverted2 = ImageChops.invert(img2)
        box1 = img1.crop(inverted1.getbbox())
        box2 = img2.crop(inverted2.getbbox())

        centerOfBox1 = (int(0.5 * box1.width), int(0.5 * box1.height))
        centerOfBox2 = (int(0.5 * box2.width), int(0.5 * box2.height))
        if box1.getpixel(centerOfBox1) == box2.getpixel(centerOfBox2):
            return False

        toBeFilled = img2 if box1.getpixel(
            centerOfBox1) == (0, 0, 0) else img1

        alreadyFilled = (img1, img2)[toBeFilled == img1]

        ImageDraw.floodfill(
            toBeFilled, xy=(0, 0), value=(255, 0, 255))

        # Make everything not magenta black
        n = np.array(toBeFilled)
        n[(n[:, :, 0:3] != [255, 0, 255]).any(2)] = [0, 0, 0]

        # Revert all artifically filled magenta pixels to white
        n[(n[:, :, 0:3] == [255, 0, 255]).all(2)] = [255, 255, 255]

        filled = Image.fromarray(n)
        if self.checkCorrelation(filled, alreadyFilled) > 0.96:
            return True
        return False

    def checkMirrored(self, axis):
        direction = (Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT)[
            axis == "vertical"]

        def performCheck(img1, img2):
            flipped = img1.transpose(direction)
            if self.checkIdentical(flipped, img2):
                return True
            return False

        return performCheck

    def checkRotated(self, deg):
        rotations = {
            90: Image.ROTATE_90,
            180: Image.ROTATE_180,
            270: Image.ROTATE_270
        }

        def performCheck(img1, img2):
            rotated = img1.transpose(rotations[deg])

            if self.checkIdentical(rotated, img2):
                return True
            return False

        return performCheck

    def checkScaled(self, img1, img2):
        return False

    def checkDeleted(self, img1, img2):
        return False

    def checkChanged(self, img1, img2):
        return True

    def horizontalTest(self, test):
        def performTest(matrix):
            if test(matrix[0], matrix[1]) & test(matrix[2], matrix[3]):
                return True
            return False
        return performTest

    def verticalTest(self, test):
        def performTest(matrix):
            if test(matrix[0], matrix[2]) & test(matrix[1], matrix[3]):
                return True
            return False
        return performTest

    def Solve(self, problem):
        print(problem.name)
        probImgs, ansImgs = self.getImages(problem)
        bestScore = float('-inf')
        bestCandidate = 1
        ansImgs = list(map(lambda tuple: tuple[1], ansImgs))

        for index, candidate in enumerate(ansImgs):
            matrix = list(map(lambda tuple: tuple[1], probImgs))
            matrix.append(candidate)
            score = 0
            for key, toPerform in self.tests.items():
                matchFound = False
                for test in toPerform:
                    testOnHorizontal = self.horizontalTest(test)
                    if testOnHorizontal(matrix):
                        score += self.weights[key]
                        matchFound = True
                        break
                if matchFound:
                    print("Horizontal match found: " + key)
                    break

            for key, toPerform in self.tests.items():
                matchFound = False
                for test in toPerform:
                    testOnVertical = self.verticalTest(test)
                    if testOnVertical(matrix):
                        score += self.weights[key]
                        matchFound = True
                        break
                if matchFound:
                    print("Vertical match found: " + key)
                    break
            print(" Candidate: " + str(index) +
                  " Old Score: " + str(bestScore) +
                  " New Score: " + str(score))
            if score > bestScore:
                print("New best candidate, old: " +
                      str(bestCandidate) + " new: " + str(index + 1))
                bestScore = score
                bestCandidate = index + 1
        print("Solution: " + str(bestCandidate))
        return bestCandidate
