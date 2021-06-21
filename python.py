 # spiral matrix II 
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = [[0] * n for i in range(n)]
        loop = n // 2
        startX = startY = 0
        count = 1
        offset = 1
        mid = n // 2
        while loop > 0:
            i = startX
            j = startY
            while j < startY + n - offset:
                res[i][j] = count
                count += 1
                j += 1
            while i < startX + n - offset:
                res[i][j] = count
                count += 1
                i += 1
            while j > startY:
                res[i][j] = count
                count += 1
                j -= 1
            while i > startX:
                res[i][j] = count
                count += 1
                i -= 1
            loop -= 1
            offset += 2
            startX += 1
            startY += 1
        
        if n % 2 == 1:
            res[mid][mid] = count
        
        return res