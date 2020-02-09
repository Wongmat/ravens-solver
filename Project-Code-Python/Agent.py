from collections import OrderedDict

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


class Agent:

    def __init__(self):
        self.tests = [
            ("Unchanged", self.checkIdentical, 6),
            ("Inverted", self.checkInverted, 5),
            ("Mirrored", self.checkMirrored, 4),
            ("Rotated", self.checkRotated, 3),
            ("Changed", self.checkChanged, 2),
            ("Deleted", self.checkDeleted, 1),
            ("Unknown", self.checkUnknown, 0),
        ]

    def blackAndWhite(self, image):
        image = image.convert('L')
        return image.point(lambda x: 0 if x < 100 else 255, '1')

    def getImages(self, problem):
        figures = problem.figures.items()
        probImgs = []
        ansImgs = []

        for figure in figures:
            key = figure[0]
            fileName = figure[1].visualFilename
            img = self.blackAndWhite(Image.open(fileName))
            if key.isalpha():
                probImgs.append((key, img))
            else:
                ansImgs.append((key, img))
        probImgs.sort(key=lambda entry: entry[0])
        ansImgs.sort(key=lambda entry: entry[0])
        return probImgs, ansImgs

    def isCorrelated(self, img1, img2):
        hist1 = img1.histogram()
        hist2 = img2.histogram()
        minima = np.minimum(hist1, hist2)
        return np.true_divide(np.sum(minima), np.sum(hist2))

    def isIdentical(self, img1, img2):
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

    def checkIdentical(self, group1, group2):
        for x in range(0, len(group1) - 1):
            group1_curr = group1[x]
            group1_next = group1[x + 1]

            group2_curr = group2[x]
            group2_next = group2[x + 1]

            if (self.isIdentical(group1_curr, group1_next) and
                    self.isIdentical(group2_curr, group2_next)):
                return True
            return False

    def isInverted(self, img1, img2):
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        if self.isCorrelated(img1, img2) > 0.96:
            return False

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
        if self.isCorrelated(filled, alreadyFilled) > 0.96:
            return True
        return False

    def checkInverted(self, group1, group2):
        for x in range(0, len(group1) - 1):
            group1_curr = group1[x]
            group1_next = group1[x + 1]

            group2_curr = group2[x]
            group2_next = group2[x + 1]

            if (self.isInverted(group1_curr, group1_next) and
                    self.isInverted(group2_curr, group2_next)):
                return True
            return False

    def isMirrored(self, img1, img2, direction):
        flipped = img1.transpose(direction)
        if self.isIdentical(flipped, img2):
            return True
        return False

    def checkMirrored(self, group1, group2):
        directions = [Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT]
        for x in range(0, len(group1) - 1):
            group1_curr = group1[x]
            group1_next = group1[x + 1]

            group2_curr = group2[x]
            group2_next = group2[x + 1]

            for direction in directions:
                if (self.isMirrored(group1_curr, group1_next, direction) and
                        self.isMirrored(group2_curr, group2_next, direction)):
                    return True
        return False

    def isRotated(self, img1, img2, rotation):
        rotated = img1.transpose(rotation)

        if self.isIdentical(rotated, img2):
            return True
        return False

    def checkRotated(self, group1, group2):
        rotations = [Image.ROTATE_90,
                     Image.ROTATE_180,
                     Image.ROTATE_270
                     ]

        for x in range(0, len(group1) - 1):
            group1_curr = group1[x]
            group1_next = group1[x + 1]

            group2_curr = group2[x]
            group2_next = group2[x + 1]

            for rotation in rotations:
                if (self.isRotated(group1_curr, group1_next, rotation) and
                        self.isRotated(group2_curr, group2_next, rotation)):
                    return True
        return False

    def checkScaled(self, img1, img2):
        return False

    def checkDeleted(self, img1, img2):
        return False

    def checkChanged(self, img1, img2):
        return False

    def checkUnknown(self, img1, img2):
        return True

    def runTestOnCols(self, test, matrix):
        group1 = [matrix[i][1]
                  for i in range(len(matrix)) if i < len(matrix) / 2]
        group2 = [matrix[i][1]
                  for i in range(len(matrix)) if i >= len(matrix) / 2]
        return test(group1, group2)

    def runTestOnRows(self, test, matrix):
        group1 = [matrix[i][1] for i in range(len(matrix)) if i % 2 == 0]
        group2 = [matrix[i][1] for i in range(len(matrix)) if i % 2 != 0]
        return test(group1, group2)

    def Solve(self, problem):
        if "Problem B" not in problem.name:
            return -1
        print(problem.name)
        probImgs, ansImgs = self.getImages(problem)
        bestScore = -99999
        bestCandidate = 1
        for index, candidate in ansImgs:
            matrix = probImgs[:]
            matrix.append((index, candidate))
            score = 0

            for subject, test, value in self.tests:
                if self.runTestOnCols(test, matrix):
                    print("Conclusion for columns: {}".format(subject))
                    score += value
                    break

            for subject, test, value in self.tests:
                if self.runTestOnRows(test, matrix):
                    print("Conclusion for rows: {}".format(subject))
                    score += value
                    break
            if score > bestScore:
                print("New best candidate: %s, with a score of %d. Old best candidate was %d with a best score of %d." %
                      (index, score, bestCandidate, (0, bestScore)[bestScore >= 0]))
                bestScore = score
                bestCandidate = int(index)

        print("Final solution for " + problem.name + ": " + str(bestCandidate))
        return bestCandidate
        # img0 = self.blackAndWhite(probImgs[0][1])
        # img1 = self.blackAndWhite(probImgs[1][1])
        # img2 = self.blackAndWhite(probImgs[2][1])
        # ans = self.blackAndWhite(ansImgs[5])
        # inverted0 = ImageChops.invert(img0)
        # inverted1 = ImageChops.invert(img1)
        # inverted2 = ImageChops.invert(img2)
        # invertedA = ImageChops.invert(ans)
        # box0 = img0.crop(inverted0.getbbox())
        # box1 = img1.crop(inverted1.getbbox())
        # box2 = img2.crop(inverted2.getbbox())
        # boxA = ans.crop(invertedA.getbbox())
        # diff1 = ImageChops.difference(img0, img1)
        # diff2 = ImageChops.difference(img2, ans)
        # diff3 = ImageChops.difference(box2, boxA)
        # diff2 = diff2.filter(ImageFilter.ModeFilter)
        # print(self.isCorrelated(diff1, diff2))
