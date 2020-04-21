import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


class Pair:
    def __init__(self, key1, img1, key2, img2):
        self.img1 = img1
        self.key1 = key1
        self.img2 = img2
        self.key2 = key2
        self.testResults = {}
        self.diffs = {}
        self.score = 0

    def __repr__(self):
        return "% s - % s | Diffs: % s" % (self.key1, self.key2, self.diffs)


class Agent:
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

    def calcDarknessRatio(self, img1, img2):
        arr = np.asarray(img1, dtype="int64")
        arr2 = np.asarray(img2, dtype="int64")
        size1 = arr.size
        size2 = arr2.size
        count1 = np.count_nonzero(arr != 1)
        count2 = np.count_nonzero(arr2 != 1)
        ratio1 = count1 / size1
        ratio2 = count2 / size2
        return ratio1 - ratio2

    def darknessRatio(self, pairs):
        for pair in pairs:
            ratio = self.calcDarknessRatio(pair.img1, pair.img2)
            pair.testResults['darkness_ratio'] = ratio
        return pairs

    def normalize(self, pairs):
        first_pair = pairs[0]
        tests = first_pair.testResults.keys()
        totals = dict.fromkeys(tests, 0)

        for pair in pairs:
            results = pair.testResults
            for test in tests:
                totals[test] += results[test]

        for pair in pairs:
            results = pair.testResults
            for test in tests:
                results[test] /= 1 if totals[test] == 0 else totals[test]
        return pairs

    def calcPixelIntersectRatio(self, img1, img2):
        union = ImageChops.logical_and(img1, img2)
        intersect = ImageChops.logical_or(img1, img2)
        union_arr = np.asarray(union, dtype="int64")
        intersect_arr = np.asarray(intersect, dtype="int64")
        intersections = np.count_nonzero(intersect_arr != 1)
        totalDark = np.count_nonzero(union_arr != 1)
        return intersections / totalDark

    def pixelIntersectRatio(self, pairs):
        for pair in pairs:
            ratio = self.calcPixelIntersectRatio(pair.img1, pair.img2)
            pair.testResults['pixel_intersect'] = ratio

        return pairs

    def getDiagonals(self, matrix):
        rows, cols = matrix.shape
        if rows < 3 & cols < 3:
            return []
        else:
            diagonals = []
            diagonals.append(matrix.diagonal())
            for x in range(1, cols - 1):
                right_diag = matrix.diagonal(offset=x)
                left_diag = matrix.diagonal(offset=(x * -1))
                diagonals.append(right_diag)
                diagonals.append(left_diag)
            return diagonals

    def getPairs(self, probImgs, ansImgs):
        probImgs = np.append(probImgs, None)
        degree = int(math.sqrt(probImgs.size))
        shape = (degree, degree)
        matrix = np.reshape(probImgs, shape)
        testPairs = {'horizontal': [], 'vertical': [], 'diagonal': []}
        vTester = None
        hTester = None
        dTester = None

        for row in matrix:
            for i in range(row.size - 1):
                key1, img1 = row[i]
                if row[i + 1] is None:
                    hTester = row[i]
                else:
                    key2, img2 = row[i + 1]
                    pair = Pair(key1, img1, key2, img2)
                    testPairs['horizontal'].append(pair)

        for col in matrix.T:
            for i in range(col.size - 1):
                key1, img1 = col[i]
                if col[i + 1] is None:
                    vTester = col[i]
                else:
                    key2, img2 = col[i + 1]
                    pair = Pair(key1, img1, key2, img2)
                    testPairs['vertical'].append(pair)

        diagonals = self.getDiagonals(matrix)
        for diag in diagonals:
            for i in range(diag.size - 1):
                key1, img1 = diag[i]
                if diag[i + 1] is None:
                    dTester = diag[i]
                else:
                    key2, img2 = diag[i + 1]
                    pair = Pair(key1, img1, key2, img2)
                    testPairs['diagonal'].append(pair)

        vKey1, vImg1 = vTester
        hKey1, hImg1 = hTester
        dKey1, dImg1 = dTester
        candidatePairs = {'horizontal': [], 'vertical': [], 'diagonal': []}

        for key2, img2 in ansImgs:
            candidatePairs['horizontal'].append(Pair(hKey1, hImg1, key2, img2))
            candidatePairs['vertical'].append(Pair(vKey1, vImg1, key2, img2))
            candidatePairs['diagonal'].append(Pair(dKey1, dImg1, key2, img2))

        if not testPairs['diagonal']:
            del candidatePairs['diagonal']
            del testPairs['diagonal']

        return testPairs, candidatePairs

    def Solve(self, problem):
        def runTests(pairs): return self.darknessRatio(
            self.pixelIntersectRatio(pairs))

        # if "Basic Problem C-03" not in problem.name:
        # return -1
        print(problem.name)
        probImgs, ansImgs = self.getImages(problem)
        testingPairs, candidatePairs = self.getPairs(probImgs, ansImgs)
        scores = dict.fromkeys([key for key, _ in ansImgs], 0)

        for direction in testingPairs.keys():
            testingList = testingPairs[direction]
            candidateList = candidatePairs[direction]
            runTests(testingList)
            runTests(candidateList)

        for direction, candidatePairList in candidatePairs.items():
            testingPairList = testingPairs[direction]
            tests = testingPairList[0].testResults.items()

            for test, _ in tests:
                total = 0

                for candidate in candidatePairList:

                    result = candidate.testResults[test]
                    candidate.diffs[test] = 0
                    for testPair in testingPairList:
                        testingResult = testPair.testResults[test]
                        diff = abs(result - testingResult)
                        candidate.diffs[test] += diff
                        total += diff

                normalized = candidatePairList[:]
                for candidate in normalized:
                    candidate.diffs[test] = 0 if total == 0 else (
                        candidate.diffs[test] / total)
                    scores[candidate.key2] += candidate.diffs[test]

                candidatePairList = normalized

        scoresList = list(scores.items())
        scoresList.sort(key=lambda entry: entry[1], reverse=False)

        print("BEST: ", scoresList[0][0], " SCORE: ", scoresList[0][1])
        print("RUNNER UP: ", scoresList[1]
              [0], " SCORE: ", scoresList[1][1])
        print(scoresList, '\n')
        return int(scoresList[0][0])
