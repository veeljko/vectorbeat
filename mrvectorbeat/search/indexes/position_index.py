from collections import defaultdict

class PositionIndex:
    def __init__(self, tokens: list[str]):
        self.index = defaultdict(dict)
        for doc_id, token in enumerate(tokens):
            for pos, term in enumerate(tokens):
                self.index[term].setdefault(doc_id, []).append(pos) 

    def find(self, s1: str, s2: str, diff: int = 1):
        docs = set(self.index.get(s1, {})) & set(self.index.get(s2, {}))

        res = []
        for d in sorted(docs):
            for px in self.index[s1][d]:
                if any(abs(px - py) <= diff for py in self.index[s2][d]):
                    res.append(d)
                    break      

        return res

    def intersection(self, l1: list[int], l2:list[int]) -> list[int]:
        p1 = 0
        p2 = 0
        res = []
        while p1 < len(l1) and p2 < len(l2):
            if l1[p1] == l2[p2]:
                res.append(l1[p1])
                p1+=1
                p2+=1
            elif l1[p1] < l2[p2]:
                p1+=1
            else:
                p2+=1
        return res

    def union(self, l1: list[int], l2:list[int]) -> list[int]:
        p1 = 0
        p2 = 0
        res = []
        while p2 < len(l2):
            while p1 < len(l1) and l1[p1] < l2[p2]:
                res.append(l1[p1])
                p1+=1
            if p1 < len(l1) and l1[p1] == l2[p2]:
                res.append(l1[p1])
                p1+=1
            else:
                res.append(l2[p2])
            p2+=1
        while p1 < len(l1):
            res.append(l1[p1])
            p1+=1

        return res

    def difference(self, l1: list[int], l2: list[int]) -> list[int]:
        p1 = 0
        p2 = 0
        l2 = self.intersection(l1, l2)
        res = []
        while p2 < len(l2):
            while p1 < len(l1) and l1[p1] < l2[p2]:
                res.append(l1[p1])
                p1+=1
            if p1 < len(l1) and l1[p1] == l2[p2]:
                p1+=1
            p2+=1
        while p1 < len(l1):
            res.append(l1[p1])
            p1+=1

        return res
