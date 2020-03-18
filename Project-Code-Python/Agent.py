import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


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

    def darknessRatio(self, img1, img2):
        arr = np.asarray(img1, dtype="int64")
        arr2 = np.asarray(img2, dtype="int64")
        size1 = arr.size
        size2 = arr2.size
        count1 = np.count_nonzero(arr != 1)
        count2 = np.count_nonzero(arr2 != 1)
        ratio1 = count1 / size1
        ratio2 = count2 / size2
        return ratio1 - ratio2

    def pixelIntersect(self, img1, img2):
        union = ImageChops.logical_and(img1, img2)
        intersect = ImageChops.logical_or(img1, img2)
        union_arr = np.asarray(union, dtype="int64")
        intersect_arr = np.asarray(intersect, dtype="int64")
        intersections = np.count_nonzero(intersect_arr != 1)
        totalDark = np.count_nonzero(union_arr != 1)
        return intersections / totalDark

    def initTests(self, staticGroups):
        viable_tests = []
        for category, test, value in self.tests:
            testTestingGroup = test(staticGroups)
            if testTestingGroup is not None:
                viable_tests.append((category, testTestingGroup, value))
        return viable_tests

    def blackAndWhite(self, image):
        image = image.convert('L')
        return image.point(lambda x: 0 if x < 100 else 255, '1')

    def getImages(self, problem):
        figures = sorted(problem.figures.items())
        probImgs = []
        ansImgs = []

        for key, obj in figures:
            fileName = obj.visualFilename
            img = self.blackAndWhite(Image.open(fileName))
            if key.isalpha():
                probImgs.append((key, img))
            else:
                ansImgs.append((key, img))

        return np.array(probImgs, dtype='U1, object'), ansImgs

    def getPairs(self, probImgs):
        probImgs = np.append(probImgs, None)
        shape = (int(math.sqrt(probImgs.size)), int(math.sqrt(probImgs.size)))
        matrix = np.reshape(probImgs, shape)
        horPairs = []
        vertPairs = []
        horTester = None
        vertTester = None
        for row in matrix:
            for i in range(row.size - 1):
                if row[i + 1] is None:
                    horTester = row[i]
                else:
                    horPairs.append((row[i], row[i + 1]))

        for col in matrix.T:
            for i in range(col.size - 1):
                if col[i + 1] is None:
                    vertTester = col[i]
                else:
                    vertPairs.append((col[i], col[i + 1]))

        return horPairs, horTester, vertPairs, vertTester

    def getRowsAndCols(self, probImgs):
        probImgs = np.append(probImgs, None)
        row_splits = np.split(probImgs, math.sqrt(probImgs.size))
        col_splits = np.transpose(row_splits)
        rows = row_splits[:-1]
        cols = col_splits[:-1]
        test_row = row_splits[-1][:-1]
        test_col = col_splits[-1][:-1]
        return rows, cols, test_row, test_col

    def getHistogramCorrelation(self, img1, img2):
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
                    if count > (rows * cols * 0.02):
                        return False
        return True

    def groupIsIdentical(self, group):
        for x in range(0, len(group) - 1):
            img1 = group[x]
            img2 = group[x + 1]
            if not self.isIdentical(img1, img2):
                return False
        return True

    def checkIdentical(self, staticGroups):
        for group in staticGroups:
            if not self.groupIsIdentical(group):
                return None

        def testTestingGroup(testingGroup):
            return self.groupIsIdentical(testingGroup)
        return testTestingGroup

    def isInverted(self, group):
        for x in range(0, len(group) - 1):
            img1 = group[x]
            img2 = group[x + 1]
            img1 = img1.convert('RGB')
            img2 = img2.convert('RGB')
            if self.getHistogramCorrelation(img1, img2) > 0.96:
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
            if self.getHistogramCorrelation(filled, alreadyFilled) < 0.96:
                return False
        return True

    def checkInverted(self, staticGroups):
        for group in staticGroups:
            if not self.isInverted(group):
                return None

        def testTestingGroup(testingGroup):
            return self.isInverted(testingGroup)
        return testTestingGroup

    def isMirrored(self, group, direction):
        for x in range(0, len(group) - 1):
            img1 = group[x]
            img2 = group[x + 1]
            flipped = img1.transpose(direction)
            if not self.isIdentical(flipped, img2):
                return False
        return True

    def checkMirrored(self, staticGroups):
        directions = [Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT]

        viable_directions = []
        for direction in directions:
            applicable = True
            for group in staticGroups:
                if not self.isMirrored(group, direction):
                    applicable = False
                    break
            if applicable:
                viable_directions.append(direction)

        if len(viable_directions) == 0:
            return None

        def testTestingGroup(testingGroup):
            for direction in viable_directions:
                if self.isMirrored(testingGroup, direction):
                    return True
            return False

        return testTestingGroup

    def isRotated(self, group, rotation):
        for x in range(0, len(group) - 1):
            img1 = group[x]
            img2 = group[x + 1]

            rotated = img1.transpose(rotation)

            if self.isIdentical(rotated, img2):
                return True
            return False

    def checkRotated(self, staticGroups):
        rotations = [Image.ROTATE_90,
                     Image.ROTATE_180,
                     Image.ROTATE_270
                     ]

        viable_rotations = []
        for rotation in rotations:
            applicable = True
            for group in staticGroups:
                if not self.isRotated(group, rotation):
                    applicable = False
                    break
            if applicable:
                viable_rotations.append(rotation)

        if len(viable_rotations) == 0:
            return None

        def testTestingGroup(testingGroup):
            for rotation in viable_rotations:
                if self.isRotated(testingGroup, rotation):
                    return True
            return False

        return testTestingGroup

    def checkScaled(self, staticGroup):
        return None

    def checkDeleted(self, staticGroup):
        return None

    def getDiff(self, img1, img2):
        diff = ImageChops.difference(img1, img2)
        return diff.filter(ImageFilter.ModeFilter)

    def checkChanged(self, staticGroups):
        for group in staticGroups:
            firstImg = group[0]
            secondImg = group[1]
            base_diff = self.getDiff(firstImg, secondImg)

            for x in range(2, len(group) - 1):
                img1 = group[x]
                img2 = group[x + 1]

                diff = self.getDiff(img1, img2)
                if self.getHistogramCorrelation(diff, base_diff) <= 0.999:
                    return None

        def testTestingGroup(testingGroup):
            for x in range(0, len(testingGroup) - 1):
                img1 = testingGroup[x]
                img2 = testingGroup[x + 1]

                diff = self.getDiff(img1, img2)
                if self.getHistogramCorrelation(diff, base_diff) >= 0.999:
                    return True
            return False

        return testTestingGroup

    def checkUnknown(self, staticGroup):
        return lambda testingGroup: True

    def Solve(self, problem):
        # if "Basic Problem C" not in problem.name:
        #    return -1
        print(problem.name)
        probImgs, ansImgs = self.getImages(problem)
        horPairs, horTester, vertPairs, vertTester = self.getPairs(probImgs)

        horTestPairs = [(self.pixelIntersect(img1[1], img2[1]), self.darknessRatio(
            img1[1], img2[1])) for (img1, img2) in horPairs]

        vertTestPairs = [(self.pixelIntersect(img1[1], img2[1]),
                          self.darknessRatio(
            img1[1], img2[1])) for (img1, img2) in vertPairs]

        horAnsPairs = [(key, self.pixelIntersect(horTester[1], candidate),
                        self.darknessRatio(horTester[1], candidate)) for (key, candidate) in ansImgs]

        vertAnsPairs = [(key, self.pixelIntersect(vertTester[1], candidate),
                         self.darknessRatio(vertTester[1], candidate)) for (key, candidate) in ansImgs]

        scores = {key: 0 for (key, image) in ansImgs}
        for (intersect, darkness) in horTestPairs:
            similarities = []
            print('horizontal')
            for (key, candidateIntersect, candidateDarkness) in horAnsPairs:
                intDiff = abs(
                    candidateIntersect - intersect)
                darkDiff = abs(
                    candidateDarkness - darkness)

                similarities.append([key, intDiff, darkDiff])

            for x in range(1, 3):
                similarities.sort(key=lambda score: score[x], reverse=True)
                for i, candidate in enumerate(similarities):
                    key = candidate[0]
                    scores[key] += i
            print(similarities)

        for (intersect, darkness) in vertTestPairs:
            similarities = []
            print('vertical')
            for (key, candidateIntersect, candidateDarkness) in vertAnsPairs:
                intDiff = abs(
                    candidateIntersect - intersect)
                darkDiff = abs(
                    candidateDarkness - darkness)

                similarities.append([key, intDiff, darkDiff])

            for x in range(1, 3):
                similarities.sort(key=lambda score: score[x], reverse=True)
                for i, candidate in enumerate(similarities):
                    key = candidate[0]
                    scores[key] += i
            print(scores)
            print(similarities)
        bestCandidates = [
            key for key, value in scores.items()
            if value == max(scores.values())]

        if len(bestCandidates) > 1:
            print(bestCandidates[1])
            return int(bestCandidates[1])

        else:
            print(bestCandidates[0])
            return int(bestCandidates[0])
        # for x in range(1, 4):
        #     scores.sort(key=lambda score: score[x], reverse=True)
        # for i, candidate in enumerate(scores):
        #     candidate[5] += i
        # print(bestCandidates[0])
        # return int(bestCandidates[0])
        # scores.sort(key=lambda score: score[5], reverse=True)
        # print(scores, scores[0][0])
        # return int(scores[0][0])
        # for key, img in ansImgs:
        #     print(key, self.darknessRatio(img))
        # bestScore = -99999
        # bestCandidate = 1
        # rows, cols, test_row, test_col = self.getRowsAndCols(probImgs)

        # horizontalTests = self.initTests(rows)
        # verticalTests = self.initTests(cols)

        # for index, candidate in ansImgs:
        #     col_to_test = list(test_col)
        #     row_to_test = list(test_row)
        #     col_to_test.append(candidate)
        #     row_to_test.append(candidate)
        #     score = 0

        #     conclusion = "Conclusion for %s:" % (index) + " "
        #     for subject, test, value in verticalTests:
        #         if test(col_to_test):
        #             conclusion += "Columns: %s" % (subject) + " "
        #             score += value
        #             break

        #     for subject, test, value in horizontalTests:
        #         if test(row_to_test):
        #             conclusion += "Rows: %s" % (subject)
        #             score += value
        #             break

        #     print(conclusion)

        #     if score > bestScore:
        #         print("New best candidate: %s, with a score of %d. " %
        #               (index, score) +
        #               " Old best candidate was %d with a best score of %d." %
        #               (bestCandidate, (0, bestScore)[bestScore >= 0]))
        #         bestScore = score
        #         bestCandidate = int(index)

        # print("Final solution for " + problem.name +
        #       ": " + str(bestCandidate) + "\n")
        # return bestCandidate
