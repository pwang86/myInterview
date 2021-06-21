public class Solution {
    // spiral matrix II
    public int[][] GenerateMatrix(int n) {
        
        int[][] res = new int[n][];
        for (int i = 0; i < n; i++) {
            res[i] = new int[n];
        }
        
        int loop = n / 2;
        int startX = 0, startY = 0;
        int offset = 1;
        int count = 1;
        int mid = n / 2;
        
        while (loop > 0) {
            int i = startX, j = startY;
            
            for (; j < startY + n - offset; j++) {
                res[i][j] = count++;
            }
            
            for(; i < startX + n - offset; i++) {
                res[i][j] = count++;
            }
            
            for(; j > startY; j--) {
                res[i][j] = count++;
            }
            
            for(; i> startX; i--) {
                res[i][j] = count++;
            }
            
            loop--;
            
            startX += 1;
            startY += 1;
            
            offset += 2;
            
        }
        
        if (n % 2 == 1) {
            res[mid][mid] = count;
        }
        
        return res;
    }
} 
