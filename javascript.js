// spiral matrix II
/**
 * @param {number} n
 * @return {number[][]}
 */
 var generateMatrix = function(n) {
  let res = Array.from({length: n}).map(() => new Array(n));
  let loop = n >> 1;
  let offset = 1;
  let count = 1;
  let startX = 0, startY = 0;
  const mid = n >> 1;
  
  while (loop > 0) {
      let i = startX, j = startY;
      while (j < startY + n - offset) {
          res[i][j] = count;
          count++;
          j++;
      }
      while (i < startX + n - offset) {
          res[i][j] = count;
          count++;
          i++;
      }
      while (j > startY) {
          res[i][j] = count;
          count++;
          j--;
      }
      while (i > startX) {
          res[i][j] = count;
          count++;
          i--;
      }
      loop--;
      startX++;
      startY++;
      offset += 2;
  }
  
  if (n % 2 == 1) {
      res[mid][mid] = count;
  }
  
  return res;
};