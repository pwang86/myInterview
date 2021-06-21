class solution {
    // spiral matrix II
    func generateMatrix(_ n: Int) -> [[Int]] {
        var res = Array(repeating: Array(repeating: 0, count: n), count: n)
        
        var loop = n / 2
        var offset = 1
        var startX = 0
        var startY = 0
        var count = 1
        var mid = n / 2
        
        while loop > 0 {
            var i = startX
            var j = startY
            while j < startY + n - offset {
                res[i][j] = count
                count += 1
                j += 1
            }
             while i < startX + n - offset {
                res[i][j] = count
                count += 1
                i += 1
            }
             while j > startY {
                res[i][j] = count
                count += 1
                j -= 1
            }
            while i > startX {
                res[i][j] = count
                count += 1
                i -= 1
            }
            offset += 2
            startX += 1
            startY += 1
            loop -= 1
        }
        if n % 2 == 1 {
            res[mid][mid] = count
        }
        return res
    }

}