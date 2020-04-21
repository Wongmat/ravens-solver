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
        return "<% s - % s | Diffs: % s | TestRes: % s>" % (self.key1, self.key2, self.diffs, self.testResults)


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
        count1 = np.count_nonzero(arr != 1)
        count2 = np.count_nonzero(arr2 != 1)
        return count1 / count2

    def darknessRatio(self, pairs):
        for pair in pairs:
            ratio = self.calcDarknessRatio(pair.img1, pair.img2)
            pair.testResults['darkness_ratio'] = ratio
        return pairs

    def calcDarkDiff(self, img1, img2):
        arr = np.asarray(img1, dtype="int64")
        arr2 = np.asarray(img2, dtype="int64")
        size1 = arr.size
        size2 = arr2.size
        count1 = np.count_nonzero(arr != 1)
        count2 = np.count_nonzero(arr2 != 1)
        ratio1 = count1 / size1
        ratio2 = count2 / size2
        return ratio1 - ratio2

    def darkDiff(self, pairs):
        for pair in pairs:
            ratio = self.calcDarkDiff(pair.img1, pair.img2)
            pair.testResults['dark_diff'] = ratio
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

    def calcNonMatchingPixelRatio(self, img1, img2):
        xor = ImageChops.logical_xor(img1, img2)
        inverted = ImageChops.invert(xor)
        inverted_arr = np.asarray(inverted, dtype="int64")
        nonmatching = np.count_nonzero(inverted_arr != 1)
        totalPixels = inverted_arr.size
        return nonmatching / totalPixels

    def nonMatchingPixelRatio(self, pairs):
        for pair in pairs:
            ratio = self.calcNonMatchingPixelRatio(pair.img1, pair.img2)
            pair.testResults['non_matching_pixel'] = ratio

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
        testPairs = {'horizontal': [], 'horizontal2': [],
                     'vertical': [], 'vertical2': [], 'diagonal': []}
        vTester = None
        v2Tester = None
        hTester = None
        h2Tester = None
        dTester = None

        for row in matrix:
            for i in range(row.size - 1):
                opposite = row.size - 1 - i
                key1, img1 = row[i]
                if row[i + 1] is None:
                    hTester = row[i]

                elif row[opposite] is None:
                    h2Tester = row[i]

                else:
                    key2, img2 = row[i + 1]
                    pair = Pair(key1, img1, key2, img2)
                    testPairs['horizontal'].append(pair)

                    if opposite > i:
                        h2_key2, h2_img2 = row[opposite]
                        h2_pair = Pair(key1, img1, h2_key2, h2_img2)
                        testPairs['horizontal2'].append(h2_pair)

        for col in matrix.T:
            for i in range(col.size - 1):
                opposite = col.size - 1 - i
                key1, img1 = col[i]
                if col[i + 1] is None:
                    vTester = col[i]

                elif col[opposite] is None:
                    v2Tester = col[i]
                else:
                    key2, img2 = col[i + 1]
                    pair = Pair(key1, img1, key2, img2)
                    testPairs['vertical'].append(pair)

                    if opposite > i:
                        v2_key2, v2_img2 = col[opposite]
                        v2_pair = Pair(key1, img1, v2_key2, v2_img2)
                        testPairs['vertical2'].append(v2_pair)

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

        candidatePairs = {'horizontal': [], 'horizontal2': [],
                          'vertical': [], 'vertical2': [], 'diagonal': []}

        for key2, img2 in ansImgs:
            candidatePairs['horizontal'].append(Pair(hKey1, hImg1, key2, img2))

            candidatePairs['vertical'].append(Pair(vKey1, vImg1, key2, img2))

            candidatePairs['diagonal'].append(Pair(dKey1, dImg1, key2, img2))

            if v2Tester:
                h2Key1, h2Img1 = h2Tester
                v2Key1, v2Img1 = v2Tester
                candidatePairs['horizontal2'].append(
                    Pair(h2Key1, h2Img1, key2, img2))

                candidatePairs['vertical2'].append(
                    Pair(v2Key1, v2Img1, key2, img2))

        for key in list(testPairs.keys())[:]:
            if not testPairs[key]:
                del candidatePairs[key]
                del testPairs[key]

        return testPairs, candidatePairs

    def Solve(self, problem):
        def runTests(pairs): return self.darkDiff(self.pixelIntersectRatio(
            self.nonMatchingPixelRatio(pairs)))

        # if "Basic Problem E-01" not in problem.name:
        #    return -1
        print(problem.name)
        probImgs, ansImgs = self.getImages(problem)
        # xor = ImageChops.logical_xor(probImgs[0][1], probImgs[1][1])
        # xor.show()
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
                # print(test, ': ')
                for candidate in candidatePairList:

                    # if (candidate.key1 == 'F') & (candidate.key2 == '1'):
                     #   self.calcDarknessRatio2(candidate.img1, candidate.img2)

                    result = candidate.testResults[test]
                    candidate.diffs[test] = 0
                    for testPair in testingPairList:
                        testingResult = testPair.testResults[test]
                        diff = abs(result - testingResult)
                        candidate.diffs[test] += diff
                        total += diff

                # print('BEFORE: ', candidatePairList, '\n')
                normalized = candidatePairList[:]
                for candidate in normalized:
                    candidate.diffs[test] = 0 if total == 0 else (
                        candidate.diffs[test] / total)
                    scores[candidate.key2] += candidate.diffs[test]

                # print('AFTER: ', normalized, '\n')
                # print('TEST PAIRS: ', testingPairList, '\n')
                candidatePairList = normalized

        scoresList = list(scores.items())
        scoresList.sort(key=lambda entry: entry[1], reverse=False)

        print("BEST: ", scoresList[0][0], " SCORE: ", scoresList[0][1])
        print("RUNNER UP: ", scoresList[1]
              [0], " SCORE: ", scoresList[1][1])
        print(scoresList, '\n')
        return int(scoresList[0][0])
