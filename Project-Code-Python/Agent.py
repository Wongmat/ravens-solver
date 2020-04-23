import math
from copy import deepcopy

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


class Pair:
    def __init__(self, elem1, pair_type, direction, elem2=None):
        self.elem1 = elem1
        self.elem2 = elem2
        self.type = pair_type
        self.direction = direction
        self.testResults = {}

    def set_elem2(self, elem):
        self.elem2 = elem

    def __repr__(self):
        return "<% s % s Pair % s - % s >" % (self.direction, self.type,
                                              self.elem1.key, self.elem2.key if self.elem2 else 'NONE')

    def run_tests(self, tests):
        for test in tests:
            name, result = test(self.elem1.img, self.elem2.img)
            self.testResults[name] = result


class Relation:
    def __init__(self, testerPair, candidatePair):
        self.testerPair = testerPair
        self.candidatePair = candidatePair
        self.diffs = []
        self.normalized_diffs = []
        self.type = testerPair.type
        self.direction = testerPair.direction

    def __repr__(self):
        return "<Relation % s - % s >" % (self.testerPair, self.candidatePair)

    def calculate_diffs(self):
        for key in self.testerPair.testResults.keys():
            tester_pair = self.testerPair.testResults[key]
            candidate_pair = self.candidatePair.testResults[key]
            diff = abs(tester_pair - candidate_pair)
            self.diffs.append((key, diff))


class Element:
    def __init__(self, key, img):
        self.key = key
        self.img = img

    def show(self):
        self.img.show()

    def __repr__(self):
        return "<% s - KEY: % s>" % (type(self).__name__, self.key)


class Candidate(Element):
    def __init__(self, key, img):
        super().__init__(key, img)
        self.score = 0
        self.pairs = []
        self.relations = []

    def __repr__(self):
        return "<% s - KEY: % s>(% s)" % (type(self).__name__, self.key, self.score)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def add_relation(self, relation):
        self.relations.append(relation)


class Tester(Element):
    def __init__(self, key, img):
        super().__init__(key, img)
        self.pairs = []


class Agent:
    def blackAndWhite(self, image):
        image = image.convert('L')
        return image.point(lambda x: 0 if x < 100 else 255, '1')

    def initElements(self, problem):
        figures = sorted(problem.figures.items())
        testers = []
        candidates = []

        for key, obj in figures:
            fileName = obj.visualFilename
            img = self.blackAndWhite(Image.open(fileName))
            if key.isalpha():
                testers.append(Tester(key, img))
            else:
                candidates.append(Candidate(key, img))

        return np.array(testers), candidates

    def classicDiagonals(self, matrix):
        lonelyTesters = []
        diagonals = []
        diagSeqs = self.getDiagonals(matrix)
        for diag in diagSeqs:
            for i in range(diag.size - 1):
                tester1 = diag[i]
                if diag[i + 1] is None:
                    lonelyTesters.append(
                        Pair(tester1, 'adjacent', 'diagonal'))
                else:
                    tester2 = diag[i + 1]
                    pair = Pair(tester1, 'adjacent', 'diagonal', tester2)
                    diagonals.append(pair)

        return lonelyTesters, diagonals

    def newDiagonals(self, matrix):
        lonelyTesters = []
        diagonals = {'adjacent': {'diagonal': []},
                     'separated': {'diagonal': []}}
        center_diag = matrix.diagonal()
        lonelyTesters.append(Pair(center_diag[0], 'separated', 'diagonal'))
        lonelyTesters.append(Pair(center_diag[1], 'adjacent', 'diagonal'))
        diagonals['adjacent']['diagonal'].append(Pair(
            center_diag[0], 'adjacent', 'diagonal', center_diag[1]))
        diag1 = matrix[[2, 0, 1], [1, -1, 0]]
        diag2 = matrix[[1, 2, 0], [-1, 0, 1]]

        for i in range(0, diag1.size):
            diag1_tester1 = diag1[i]
            diag2_tester1 = diag2[i]
            for j in range(i + 1, diag1.size):
                diag1_tester2 = diag1[j]
                diag2_tester2 = diag2[j]
                pair_type = 'adjacent' if j == i + 1 else 'separated'
                diag1_pair = Pair(diag1_tester1, pair_type,
                                  'diagonal', diag1_tester2)
                diag2_pair = Pair(diag2_tester1, pair_type,
                                  'diagonal', diag2_tester2)
                diagonals[pair_type]['diagonal'].append(diag1_pair)
                diagonals[pair_type]['diagonal'].append(diag2_pair)

        diag3 = matrix[[1, 2, 0], [2, 1, 0]]
        diag4 = [[2, 0, 1], [0, 2, 1]]
        diag5 = [[0, 1, 2], [1, 0, 2]]
        print(diag3, diag4, diag5)

        return lonelyTesters, diagonals

    def getTesterPairs(self, testers):
        probImgs = np.append(testers, None)
        degree = int(math.sqrt(probImgs.size))
        shape = (degree, degree)
        matrix = np.reshape(probImgs, shape)

        testerPairs = {'adjacent': {}}
        lonelyTesters = []

        if degree > 2:
            lonelyDiagTesters, diagonals = self.newDiagonals(matrix)
            testerPairs = diagonals
            lonelyTesters = lonelyDiagTesters

        adjacent = testerPairs['adjacent']
        adjacent['horizontal'] = []
        for row in matrix:
            for i in range(row.size - 1):
                opposite = row.size - 1 - i
                tester1 = row[i]
                if row[i + 1] is None:
                    lonelyTesters.append(
                        Pair(tester1, 'adjacent', 'horizontal'))

                elif row[opposite] is None and degree > 2:
                    lonelyTesters.append(
                        Pair(tester1,
                             'separated', 'horizontal'))

                else:
                    tester2 = row[i + 1]
                    pair = Pair(tester1, 'adjacent', 'horizontal', tester2)
                    adjacent['horizontal'].append(pair)

                    if opposite > i and degree > 2:
                        separated = testerPairs['separated']
                        sep_tester = row[opposite]
                        sep_pair = Pair(tester1, 'separated',
                                        'horizontal', sep_tester)

                        if 'horizontal' not in separated:
                            separated['horizontal'] = []
                        separated['horizontal'].append(sep_pair)

        adjacent['vertical'] = []

        for col in matrix.T:
            for i in range(col.size - 1):
                opposite = col.size - 1 - i
                tester1 = col[i]
                if col[i + 1] is None:
                    lonelyTesters.append(
                        Pair(tester1, 'adjacent', 'vertical'))

                elif col[opposite] is None and degree > 2:
                    lonelyTesters.append(
                        Pair(tester1, 'separated', 'vertical'))

                else:
                    tester2 = col[i + 1]
                    pair = Pair(tester1, 'adjacent', 'vertical', tester2)
                    adjacent['vertical'].append(pair)

                    if opposite > i and degree > 2:
                        separated = testerPairs['separated']
                        sep_tester = col[opposite]
                        sep_pair = Pair(tester1, 'separated',
                                        'vertical', sep_tester)

                        if 'vertical' not in separated:
                            separated['vertical'] = []

                        separated['vertical'].append(sep_pair)

        return lonelyTesters, testerPairs

    def calcDarknessRatio(self, img1, img2):
        arr = np.asarray(img1, dtype="int64")
        arr2 = np.asarray(img2, dtype="int64")
        count1 = np.count_nonzero(arr != 1)
        count2 = np.count_nonzero(arr2 != 1)
        if count2 == 0:
            count2 = 1
        return ('darkness_ratio', count1 / count2)

    def calcDarkDiff(self, img1, img2):
        arr = np.asarray(img1, dtype="int64")
        arr2 = np.asarray(img2, dtype="int64")
        size1 = arr.size
        size2 = arr2.size
        count1 = np.count_nonzero(arr != 1)
        count2 = np.count_nonzero(arr2 != 1)
        ratio1 = count1 / size1
        ratio2 = count2 / size2
        return ('dark_diff', ratio1 - ratio2)

    def calcPixelIntersectRatio(self, img1, img2):
        union = ImageChops.logical_and(img1, img2)
        intersect = ImageChops.logical_or(img1, img2)
        union_arr = np.asarray(union, dtype="int64")
        intersect_arr = np.asarray(intersect, dtype="int64")
        intersections = np.count_nonzero(intersect_arr != 1)
        totalDark = np.count_nonzero(union_arr != 1)
        if totalDark == 0:
            totalDark = 1
        return ('pixel_intersect', intersections / totalDark)

    def calcNonMatchingPixelRatio(self, img1, img2):
        xor = ImageChops.logical_xor(img1, img2)
        inverted = ImageChops.invert(xor)
        inverted_arr = np.asarray(inverted, dtype="int64")
        nonmatching = np.count_nonzero(inverted_arr != 1)
        totalPixels = inverted_arr.size
        return ('non_matching_pixel', nonmatching / totalPixels)

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

    def Solve(self, problem):
        tests = [self.calcNonMatchingPixelRatio,
                 self.calcDarknessRatio, self.calcPixelIntersectRatio]

        # if "Basic Problem B-01" not in problem.name:
        #    return -1
        print(problem.name)
        testers, candidates = self.initElements(problem)
        lonelyTesters, testerPairs = self.getTesterPairs(testers)

        totalsForNorm = dict.fromkeys(list(testerPairs.values())[0].keys(), {})

        for candidate in candidates:
            for pair in lonelyTesters:
                pair_copy = deepcopy(pair)
                pair_copy.set_elem2(candidate)
                pair_copy.run_tests(tests)
                candidate.add_pair(pair_copy)

        for entry in list(testerPairs.values()):
            for pairs in list(entry.values()):
                for pair in pairs:
                    pair.run_tests(tests)

        for candidate in candidates:
            for pair in candidate.pairs:

                tester_pairs = testerPairs[pair.type][pair.direction]

                for tester_pair in tester_pairs:
                    relation = Relation(tester_pair, pair)
                    relation.calculate_diffs()

                    for test_name, diff in relation.diffs:
                        if test_name not in totalsForNorm[pair.direction]:
                            totalsForNorm[pair.direction][test_name] = 0

                        totalsForNorm[pair.direction][test_name] += diff

                    candidate.add_relation(relation)

        for candidate in candidates:
            for relation in candidate.relations:
                for test_name, diff in relation.diffs:
                    total = totalsForNorm[relation.direction][test_name]
                    norm_score = 0 if total == 0 else diff / total
                    candidate.score += norm_score
                    normalized = (test_name, norm_score)
                    relation.normalized_diffs.append(normalized)

        candidates.sort(key=lambda candidate: candidate.score, reverse=False)

        print("BEST: ", candidates[0].key, " SCORE: ", candidates[0].score)
        print("RUNNER UP: ", candidates[1].key,
              " SCORE: ", candidates[1].score)
        print("Scores Table: ")
        print(candidates, '\n')

        return int(candidates[0].key)
