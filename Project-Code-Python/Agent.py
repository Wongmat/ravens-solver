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

    def getPairs(self, probImgs, ansImgs):
        probImgs = np.append(probImgs, None)
        degree = int(math.sqrt(probImgs.size))
        shape = (degree, degree)
        matrix = np.reshape(probImgs, shape)
        testPairs = {'horizontal': [], 'vertical': []}
        vTester = None
        hTester = None

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

        vKey1, vImg1 = vTester
        hKey1, hImg1 = hTester
        candidatePairs = {'horizontal': [], 'vertical': []}

        for key2, img2 in ansImgs:
            candidatePairs['horizontal'].append(Pair(hKey1, hImg1, key2, img2))
            candidatePairs['vertical'].append(Pair(vKey1, vImg1, key2, img2))

        return testPairs, candidatePairs

    def Solve(self, problem):
        def runTests(pairs): return self.darknessRatio(
            self.pixelIntersectRatio(pairs))

       #  if "Basic Problem B" not in problem.name and "Basic Problem C" not in problem.name:
        #     return -1
        print(problem.name)
        probImgs, ansImgs = self.getImages(problem)
        testingPairs, candidatePairs = self.getPairs(probImgs, ansImgs)
        scores = {key: 0 for key, img in ansImgs}

        for direction in testingPairs.keys():
            testingList = testingPairs[direction]
            candidateList = candidatePairs[direction]
            runTests(testingList)
            runTests(candidateList)

        for direction, testingPairList in testingPairs.items():
            candidatePairList = candidatePairs[direction]

            for testPair in testingPairList:
                print(direction, ': ', testPair.key1, '-', testPair.key2, '\n')
                for testName, result in testPair.testResults.items():
                    for candidate in candidatePairList:
                        candidateResult = candidate.testResults[testName]
                        diff = abs(result - candidateResult)
                        candidate.diffs[testName] = diff

                    candidatePairList.sort(
                        key=lambda pair: pair.diffs[testName], reverse=True)

                    for index, pair in reversed(list(
                            enumerate(candidatePairList))):
                        scores[pair.key2] += 0.75 * \
                            index if testName == 'darkness_ratio' else index

            scoresList = list(scores.items())
            scoresList.sort(key=lambda entry: entry[1], reverse=True)

        print("BEST: ", scoresList[0][0], " SCORE: ", scoresList[0][1])
        print("RUNNER UP: ", scoresList[1]
              [0], " SCORE: ", scoresList[1][1])
        print(scoresList, '\n')
        return int(scoresList[0][0])
