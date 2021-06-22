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

// Remove Linked List Elements
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} val
 * @return {ListNode}
 */
 var removeElements = function(head, val) {
    if (head == null) return head;
    
    var dummy = new ListNode(-1, head);
    var pre = dummy;
    var cur = head;
    
    while (cur != null) {
        if (cur.val == val) {
            pre.next = cur.next;
        } else {
            pre = cur;
        }
        cur = cur.next;
    }
    
    return dummy.next;
};