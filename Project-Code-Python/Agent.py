from collections import OrderedDict

import numpy
from PIL import Image, ImageChops, ImageFilter


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().

    def __init__(self):
        self.weights = {"Unchanged": 5,
                        "Mirrored": 4,
                        "Rotated": 3,
                        "Scaled": 2,
                        "Deleted": 1,
                        "Changed": 0,
                        }

        self.tests = OrderedDict([("Unchanged", [self.isIdentical]),
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

        # The primary method for solving incoming Raven's Progressive Matrices.
        # For each problem, your Agent's Solve() method will be called. At the
        # conclusion of Solve(), your Agent should return an int representing its
        # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
        # are also the Names of the individual RavensFigures, obtained through
        # RavensFigure.getName(). Return a negative number to skip a problem.
        #
        # Make sure to return your answer *as an integer* at the end of Solve().
        # Returning your answer as a string may cause your program to crash.
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

    def isIdentical(self, img1, img2):
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

    def checkMirrored(self, axis):
        direction = (Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT)[
            axis == "vertical"]

        def performCheck(img1, img2):
            flipped = img1.transpose(direction)
            if self.isIdentical(flipped, img2):
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

            if self.isIdentical(rotated, img2):
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
                    break

            if score > bestScore:
                bestScore = score
                bestCandidate = index + 1
        return bestCandidate
