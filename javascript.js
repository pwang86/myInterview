// spiral matrix II
/**
 * @param {number} n
 * @return {number[][]}
 */
var generateMatrix = function (n) {
    let res = Array.from({ length: n }).map(() => new Array(n));
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
var removeElements = function (head, val) {
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

// reverse linked list
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
 var reverseList = function(head) {
    let tmp = null;
    let pre = null;
    let cur = head;
    while (cur != null) {
        tmp = cur.next;
        cur.next = pre;
        pre = cur;
        cur = tmp;
    }
    return pre;
};


// swap nodes in pairs
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
 var swapPairs = function(head) {
    if (head == null || head.next == null) return head;
    
    let dummy = new ListNode(0, head);
    let pre = dummy;
    let cur = head;
    while (cur != null && cur.next != null) {
        let tmp = cur.next.next;
        pre.next = cur.next;
        cur.next.next = cur;
        cur.next = tmp;
        pre = cur;
        cur = cur.next;
    }
    return dummy.next;
};
// remove nth node from end of list
/**
 * @param {ListNode} head
 * @param {number} n
 * @return {ListNode}
 */
 var removeNthFromEnd = function(head, n) {
    let dummy = new ListNode(0, head);
    let slow = dummy;
    let fast = dummy;
    while (n > 0) {
        fast = fast.next;
        n--;
    }
    fast = fast.next;
    while (fast != null) {
        slow = slow.next;
        fast = fast.next;
    }
    slow.next = slow.next.next;
    return dummy.next;
};

// intersection of two linked lists
/**
 * @param {ListNode} headA
 * @param {ListNode} headB
 * @return {ListNode}
 */
 var getIntersectionNode = function(headA, headB) {
    let lenA = 0, lenB = 0;
    let curA = headA, curB = headB;
    while (curA != null) {
        curA = curA.next;
        lenA++;
    }
    while (curB != null) {
        curB = curB.next;
        lenB++;
    }
    curA = headA;
    curB = headB;
    
    if (lenB > lenA) {
        [lenA, lenB] = [lenB, lenA];
        [curA, curB] = [curB, curA];
    }
    let gap = lenA - lenB;
    while (gap > 0) {
        gap--;
        curA = curA.next;
    }
    while (curA != null) {
        if (curA == curB) {
            return curA;
        }
        curA = curA.next;
        curB = curB.next;
    }
    return null;
};

// linked list cycle II
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
 var detectCycle = function(head) {
    let fast = head;
    let slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
        if (fast == slow) {
            let index1 = fast;
            let index2 = head;
            while (index1 != index2) {
                index1 = index1.next;
                index2 = index2.next;
            }
            return index1;
        }
    }
    return null;
};

// valid anagram
/**
 * @param {string} s
 * @param {string} t
 * @return {boolean}
 */
 var isAnagram = function(s, t) {
    if (s.length != t.length)
        return false;
    let record = new Array(26).fill(0);
    for (let i of s) {
        record[i.charCodeAt() - 'a'.charCodeAt()] += 1;
    }
    for (let i of t) {
        record[i.charCodeAt() - 'a'.charCodeAt()] -= 1;
    }
    for (let i = 0; i < 26; i++) {
        if (record[i] != 0)
            return false;
    }
    return true;
};

// intersection of two arrays
/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number[]}
 */
 var intersection = function(nums1, nums2) {
    if (nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0) {
        return [];
    }
    let set1 = new Set(nums1);
    let res = new Set();

    for (let i of nums2) {
        if (set1.has(i)) {
            res.add(i);
        }
    }
    return Array.from(res);
};

// happy number
/**
 * @param {number} n
 * @return {boolean}
 */
 var isHappy = function(n) {
    let mySet = new Set();
    while (n != 1 && !mySet.has(n)) {
        mySet.add(n);
        n = getSum(n);
    }
    return n == 1;
};

function getSum(n) {
    let sum = 0;
    while (n > 0) {
        let tmp = n % 10;
        sum += tmp * tmp;
        n = parseInt(n / 10);
    }
    return sum;
} 

// two sum
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
 var twoSum = function(nums, target) {
    let res = new Array(2);
    let myMap = new Map();
    
    for (let i = 0; i < nums.length; i++) {
        let tmp = target - nums[i];
        if (myMap.has(tmp)) {
            return [i, myMap.get(tmp)];
        }
        myMap.set(nums[i], i);
    }
    return res;
};

// 4 sum II
/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @param {number[]} nums3
 * @param {number[]} nums4
 * @return {number}
 */
 var fourSumCount = function(nums1, nums2, nums3, nums4) {
    let myMap = new Map()
    let res = 0
    for (const i of nums1) {
        for (const j of nums2) {
            myMap.set(i + j, (myMap.get(i + j) || 0) + 1);
        }
    }
    
    for (const i of nums3) {
        for (const j of nums4) {
            if (myMap.has(0 - i - j)) {
                res += myMap.get(0 - i - j);
            }
        }
    }
    return res;
};

// ransome note
/**
 * @param {string} ransomNote
 * @param {string} magazine
 * @return {boolean}
 */
 var canConstruct = function(ransomNote, magazine) {
    let record = new Array(26).fill(0);
    let tmp = 0;
    for (const i of magazine) {
        tmp = i.charCodeAt() - 'a'.charCodeAt();
        record[tmp]++;
    }
    for (const i of ransomNote) {
        tmp = i.charCodeAt() - 'a'.charCodeAt();
        if (record[tmp] > 0) {
            record[tmp]--;
        } else {
            return false;
        }
    }
    return true;
};


// 3 sum
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var threeSum = function(nums) {
    let res = [];
    nums.sort((a, b) => a- b);
    
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] > 0) return res;
        if (i > 0 && nums[i - 1] == nums[i]) continue;
        
        let left = i + 1;
        let right = nums.length - 1;
        while (right > left) {
            let tmp = nums[i] + nums[left] + nums[right];
            if (tmp > 0) {
                right--;
            } else if (tmp < 0) {
                left++;
            } else {
                res.push([nums[i], nums[left], nums[right]]);
                while (right > left && nums[right] == nums[right - 1]) right--;
                while (right > left && nums[left] == nums[left + 1]) left++;
                left++;
                right--;
            }
        }
    }
    return res;
};

// 4sum
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[][]}
 */
 var fourSum = function(nums, target) {
    let res = [];
    
    nums.sort((a, b) => a - b);
    for (let i = 0; i < nums.length; i++) {
        if (i > 0 && nums[i - 1] == nums[i]) continue;
        for (let j = i + 1; j < nums.length; j++) {
            if ((j > i + 1) && nums[j - 1] == nums[j]) continue;
            let left = j + 1;
            let right = nums.length - 1;
            while (right > left) {
                let sum = nums[i] + nums[j] + nums[left] + nums[right];
                if (sum > target) {
                    right--;
                } else if (sum < target) {
                    left++;
                } else {
                    res.push([nums[i], nums[j], nums[left], nums[right]]);
                    while (right > left && nums[right - 1] == nums[right]) right--;
                    while (right > left && nums[left] == nums[left + 1]) left++;
                    left++;
                    right--;
                }
            }
        }
    }
    return res;
};

// reverse string
/**
 * @param {character[]} s
 * @return {void} Do not return anything, modify s in-place instead.
 */
 var reverseString = function(s) {
    let l = 0, r = s.length - 1;
    while (l < r) {
        [s[l], s[r]] = [s[r], s[l]];
        r--;
        l++;
    }
};

// reverse string II
/**
 * @param {string} s
 * @param {number} k
 * @return {string}
 */
 var reverseStr = function(s, k) {
    let tmp = s.split("");
    for (let i = 0; i < tmp.length; i += 2 * k) {
        let start = i, end = Math.min(start + k - 1, tmp.length - 1);
        while (start < end) {
            [tmp[start], tmp[end]] = [tmp[end], tmp[start]];
            start++;
            end--;
        }
    }
    return tmp.join("");
};

// implement strStr
/**
 * @param {string} haystack
 * @param {string} needle
 * @return {number}
 */
 var strStr = function(haystack, needle) {
    if (needle.length == 0) return 0;
    let next = new Array(needle.length);
    getNext(next, needle);
    let j = -1;
    for (let i = 0; i < haystack.length; i++) {
        while (j >= 0 && haystack[i] != needle[j + 1]) {
            j = next[j];
        }
        if (haystack[i] == needle[j + 1]) {
            j++;
        }
        if (j == needle.length - 1) {
            return (i - needle.length + 1);
        }
    }
    return -1;
};

function getNext(next, s) {
    let j = -1;
    next[0] = j;
    for (let i = 1; i < s.length; i++) {
        while (j >= 0 && s[i] != s[j + 1]) {
            j = next[j];
        }
        if (s[i] == s[j + 1]) {
            j++;
        }
        next[i] = j;
    }
}

// reverse words in a string
/**
 * @param {string} s
 * @return {string}
 */
 var reverseWords = function(s) {
    let arr = removeSpace(s).split("");
    reverseString(arr, 0, arr.length - 1);
    reverseEachWord(arr);
    return arr.join("");
};
function removeSpace(s) {
    let l = 0, r = s.length - 1;
    while (l <= r && s[l] == ' ') l++;
    while (l <= r && s[r] == ' ') r--;
    let res = "";
    while (l <= r) {
        if (s[l] != ' ' || res[res.length - 1] != ' ') {
            res += s[l];   
        }
        l++;
    }
    return res;
}
function reverseString(arr, start, end) {
    while (start < end) {
        [arr[start], arr[end]] = [arr[end], arr[start]];
        start++;
        end--;
    }
}
function reverseEachWord(arr) {
    let start = 0, end = 0;
    while (start < arr.length) {
        while (end < arr.length && arr[end] != ' ') {
            end++;
        }
        reverseString(arr, start, end - 1);
        start = end + 1;
        end++;
    }
}

// Repeated Substring Pattern
/**
 * @param {string} s
 * @return {boolean}
 */
 var repeatedSubstringPattern = function(s) {
    const len = s.length;
    if (len == 0) return false;
    
    var next = new Array(len);
    getNext(next, s);
    
    if (next[len - 1] != -1 && len % (len - (next[len - 1] + 1)) == 0) 
        return true;
    return false;
};
function getNext(next, s) {
    var j = -1
    next[0] = j
    for (let i = 1; i < s.length; i++) {
        while (j >= 0 && s[i] != s[j + 1]) {
            j = next[j];
        }
        if (s[i] == s[j + 1]) {
            j++;
        }
        next[i] = j;
    }
}

// Binary Tree Preorder Traversal
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var preorderTraversal = function(root) {
    let res = [];
    if (root == null) return res;
    
    // preOrder(root, res);

    // iterative way
    let stack = [];
    stack.push(root);
    while (stack.length != 0) {
        let tmp = stack.pop();
        res.push(tmp.val);
        if (tmp.right != null) stack.push(tmp.right);
        if (tmp.left != null) stack.push(tmp.left);
    }
    return res;
};
var preOrder = (root, res) => {
    if (root == null) return;
    res.push(root.val);
    preOrder(root.left, res);
    preOrder(root.right, res);
};


// postorder traversal
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var postorderTraversal = function(root) {
    let res = [];
    if (root == null) return res;
    // helper(root, res);
    let stack = [];
    stack.push(root);
    while (stack.length != 0) {
        let tmp = stack.pop();
        res.push(tmp.val);
        if (tmp.left != null) {
            stack.push(tmp.left);
        }
        if (tmp.right != null) {
            stack.push(tmp.right);
        }
    }
    
    return res.reverse();
};
var helper = (root, res) => {
    if (root == null) {
        return;
    } else {
        helper(root.left, res);
        helper(root.right, res);
        res.push(root.val);
    }
};

// Inorder Traversal
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var inorderTraversal = function(root) {
    let res = [];
    if (root == null) return res;
    
    // inOrder(root, res);
    let stack = [];
    let cur = root;
    while (stack.length != 0 || cur != null) {
        if (cur != null) {
            stack.push(cur);
            cur = cur.left;
        } else {
            cur = stack.pop();
            res.push(cur.val);
            cur = cur.right;
        }
    }
    return res;
};
var inOrder = (root, res) => {
    if (root == null) return;
    inOrder(root.left, res);
    res.push(root.val);
    inOrder(root.right, res);
};

// Binary Tree Level Order Traversal
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
 var levelOrder = function(root) {
    let res = [];
    if (root == null) return res;
    
    let queue = [];
    queue.push(root);
    while (queue.length != 0) {
        let size = queue.length;
        let tmp = [];
        for (let i = 0; i < size; i++) {
            let cur = queue.shift();
            tmp.push(cur.val);
            if (cur.left != null) queue.push(cur.left);
            if (cur.right != null) queue.push(cur.right);
        }
        res.push(tmp);
    }
    return res;
};

// Binary Tree Level Order Traversal II.
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
 var levelOrderBottom = function(root) {
    let res = [];
    if (root == null) return res;
    
    let dfs = (root, level) => {
        if (level >= res.length) {
            res[level] = [];
        }
        res[level].push(root.val);
        if (root.left != null) {
            dfs(root.left, level + 1);
        }
        if (root.right != null) {
            dfs(root.right, level + 1);
        }
    };
    dfs(root, 0);
    return res.reverse();
};

// Binary Tree Right Side View
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var rightSideView = function(root) {
    let res = [];
    if (root == null) return res;
    
    let queue = [];
    queue.push(root);
    while (queue.length > 0) {
        let size = queue.length;
        for (let i = 0; i < size; i++) {
            let cur = queue.shift();
            if (cur.left != null) queue.push(cur.left);
            if (cur.right != null) queue.push(cur.right);
            if (i == size - 1) res.push(cur.val);
        }
    }
    return res;
};

// Average of Levels in Binary Tree
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var averageOfLevels = function(root) {
    let res = [];
    if (root == null) return res;
    
    let queue = [];
    queue.push(root);
    while (queue.length > 0) {
        let size = queue.length;
        let sum = 0;
        for (let i = 0; i < size; i++) {
            let cur = queue.shift();
            sum += cur.val;
            if (cur.left != null) queue.push(cur.left);
            if (cur.right != null) queue.push(cur.right);
        }
        res.push(sum / size);
    }
    
    return res;
};

// Populating Next Right Pointers in Each Node
/**
 * @param {Node} root
 * @return {Node}
 */
 var connect = function(root) {
    if (root == null) {
        return root;
    }
    
    let queue = [];
    queue.push(root);
    let pre = null;
    let cur = null;
    while (queue.length > 0) {
        let size = queue.length;
        for (let i = 0; i < size; i++) {
            if (i == 0) {
                pre = queue.shift();
                cur = pre;
            } else {
                cur = queue.shift();
                pre.next = cur;
                pre = pre.next;
            }
            if (cur.left != null) queue.push(cur.left);
            if (cur.right != null) queue.push(cur.right);
        }
        pre.next = null;
    }
    return root;
};

// Invert Binary Tree
var invertTree = function(root) {
    if (root == null)
        return root;
    
    let queue = [];
    queue.push(root);
    while (queue.length > 0) {
        let size = queue.length;
        for (let i = 0; i < size; i++) {
            let cur = queue.shift();
            [cur.left, cur.right] = [cur.right, cur.left];
            if (cur.left) queue.push(cur.left);
            if (cur.right) queue.push(cur.right);
        }
    }
    return root;
};

// recursive way
var invertTree = function(root) {
    if (root == null)
        return root;
    
    [root.left, root.right] = [root.right, root.left];
    invertTree(root.left);
    invertTree(root.right);
    return root;
};

// Symmetric Tree
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
 var isSymmetric = function(root) {
    if (root == null) return true;
    return helper(root.left, root.right);
};
function helper(l, r) {
    if (l == null && r == null) return true;
    else if (l != null && r == null) return false;
    else if (l == null && r != null) return false;
    else if (l.val != r.val) return false;
    else {
        let inner = helper(l.right, r.left);
        let outer = helper(l.left, r.right);
        return inner && outer;
    }
}
// using queue in Symmetric Tree
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
 var isSymmetric = function(root) {
    if (root == null) return true;
    
    let queue = [];
    queue.push(root.left);
    queue.push(root.right);
    while (queue.length > 0) {
        let l = queue.shift();
        let r = queue.shift();
        
        if (!l && !r) continue;
        if (!l || !r || (l.val != r.val)) {
            return false;
        }
        queue.push(l.left);
        queue.push(r.right);
        queue.push(l.right);
        queue.push(r.left);
    }
    return true;
};


// Maximum Depth of Binary Tree
/**
 * @param {TreeNode} root
 * @return {number}
 */
 var maxDepth = function(root) {
    if (root == null) return 0;
    let res = 0;
    
    let queue = [];
    queue.push(root);
    while (queue.length > 0) {
        let size = queue.length;
        for (let i = 0; i < size; i++) {
            let cur = queue.shift();
            if (cur.left != null) queue.push(cur.left);
            if (cur.right != null) queue.push(cur.right);
        }
        res++;
    }
    return res;
};

// Maximum Depth of N-ary Tree
/**
 * @param {Node|null} root
 * @return {number}
 */
 var maxDepth = function(root) {
    if (root == null)
        return 0;
    return helper(root);
};

function helper(root) {
    if (root == null) return 0;
    
    let res = 0;
    for (let node of root.children) {
        res = Math.max(res, helper(node));
    }
    
    return res + 1;
}

// Minimum Depth of Binary Tree
/**
 * @param {TreeNode} root
 * @return {number}
 */
 var minDepth = function(root) {
    if (root == null) return 0;
    let res = 0;
    let queue = [];
    queue.push(root);
    while (queue.length > 0) {
        let size = queue.length;
        res++;
        for (let i = 0; i < size; i++) {
            let cur = queue.shift();
            if (cur.left == null && cur.right == null) {
                return res;
            }
            if (cur.left != null) {
                queue.push(cur.left);
            }
            if (cur.right != null) {
                queue.push(cur.right);
            }
        }
    }
    return res;
    // postorder way
    // if (root == null) return 0;
    // return helper(root);
};
function helper(root) {
    if (root == null) return 0;
    
    let leftNode = root.left;
    let rightNode = root.right;
    if (leftNode == null && rightNode != null) {
        return 1 + helper(rightNode);
    }
    if (leftNode != null && rightNode == null) {
        return 1 + helper(leftNode);
    }
    let res = 1 + Math.min(helper(leftNode), helper(rightNode));
    return res;
}

// Count Complete Tree Nodes
/**
 * @param {TreeNode} root
 * @return {number}
 */
 var countNodes = function(root) {
    if (root == null) {
        return 0;
    }
    let queue = [];
    queue.push(root);
    let res = 0;
    
    while (queue.length > 0) {
        let size = queue.length;
        for (let i = 0; i < size; i++) {
            let cur = queue.shift();
            if (cur.left != null) queue.push(cur.left);
            if (cur.right != null) queue.push(cur.right);
        }
        res += size;
    }
    return res;
    /**
     * O(logn * logn)
      
    if (root == null) {
        return 0;
    }
    let leftNode = root.left;
    let rightNode = root.right;
    let leftHeight = 0, rightHeight = 0;
    while (leftNode != null) {
        leftNode = leftNode.left;
        leftHeight++;
    }
    while (rightNode != null) {
        rightNode = rightNode.right;
        rightHeight++;
    }
    if (leftHeight == rightHeight) {
        return Math.pow(2, leftHeight + 1) - 1
    }
    return countNodes(root.left) + countNodes(root.right) + 1;
    */ 

};

// Binary Tree Paths
/**
 * @param {TreeNode} root
 * @return {string[]}
 */
 var binaryTreePaths = function(root) {
    let res = [];
    if (root == null) {
        return res;
    }
    let path = [];
    traversal(root, path, res);
    return res;
};

function traversal(root, path, res) {
    path.push(root.val);
    if (root.left == null && root.right == null) {
        let tmp = "";
        for (let i = 0; i < path.length - 1; i++) {
            tmp += path[i];
            tmp += "->";
        }
        tmp += path[path.length - 1];
        res.push(tmp);
        return;
    }
    if (root.left != null) {
        traversal(root.left, path, res);
        path.pop();
    }
    if (root.right != null) {
        traversal(root.right, path, res);
        path.pop();
    }
}

// Sum of Left Leaves
/**
 * @param {TreeNode} root
 * @return {number}
 */
 var sumOfLeftLeaves = function(root) {
    if (root == null) {
        return 0;
    }
    
    let leftValue = sumOfLeftLeaves(root.left);
    let rightValue = sumOfLeftLeaves(root.right);
    let curValue = 0;
    if (root.left != null && root.left.left == null && root.left.right == null) {
        curValue = root.left.val;
    }
    let sum = curValue + leftValue + rightValue;
    return sum;
};

// Path Sum
/**
 * @param {TreeNode} root
 * @param {number} targetSum
 * @return {boolean}
 */
 var hasPathSum = function(root, targetSum) {
    if (root == null) return false;
    return traversal(root, targetSum - root.val);
};
function traversal(root, sum) {
    if (root.left == null && root.right == null && sum == 0) return true;
    if (root.left == null && root.right == null) return false;
    
    if (root.left != null) {
        if (traversal(root.left, sum - root.left.val)) {
            return true;
        }
    }
    if (root.right != null) {
        if (traversal(root.right, sum - root.right.val)) {
            return true;
        } 
    }
    return false;
}

// path sum II
/**
 * @param {TreeNode} root
 * @param {number} targetSum
 * @return {number[][]}
 */
 var pathSum = function(root, targetSum) {
    let res = [];
    if (root == null) return res;
    let path = [];
    let preorder = (cur, sum) => {
        path.push(cur.val);
        if (!cur.left && !cur.right) {
            if (sum - cur.val == 0) {
                res.push([...path]);
            }
            return;
        }
        if (cur.left) {
            preorder(cur.left, sum - cur.val);
            path.pop();
        }
        if (cur.right) {
            preorder(cur.right, sum - cur.val);
            path.pop();
        }
    };
    preorder(root, targetSum);
    return res;
};

// Construct Binary Tree from Inorder and Postorder Traversal
/**
 * @param {number[]} inorder
 * @param {number[]} postorder
 * @return {TreeNode}
 */
 var buildTree = function(inorder, postorder) {
    if (postorder.length == 0) return null;
    
    let rootVal = postorder[postorder.length - 1];
    let index = inorder.indexOf(rootVal);
    let root = new TreeNode(rootVal);
    
    let leftInorder = inorder.slice(0, index);
    let rightInorder = inorder.slice(index + 1, inorder.length);
    
    let leftPostorder = postorder.slice(0, leftInorder.length);
    let rightPostorder = postorder.slice(leftInorder.length, postorder.length - 1);
    
    root.left = buildTree(leftInorder, leftPostorder);
    root.right = buildTree(rightInorder, rightPostorder);
    
    return root;
};

 // Construct Binary Tree from Preorder and Inorder Traversal
 /**
 * @param {number[]} preorder
 * @param {number[]} inorder
 * @return {TreeNode}
 */
var buildTree = function(preorder, inorder) {
    if (preorder.length == 0) return null;
    
    let rootVal = preorder[0];
    let rootIndex = inorder.indexOf(rootVal);
    
    let root = new TreeNode(rootVal);
    
    let leftInorder = inorder.slice(0, rootIndex);
    let rightInorder = inorder.slice(rootIndex + 1, inorder.length);
    
    let leftPreorder = preorder.slice(1, rootIndex + 1);
    let rightPreorder = preorder.slice(rootIndex + 1, preorder.length);
    
    root.left = buildTree(leftPreorder, leftInorder);
    root.right = buildTree(rightPreorder, rightInorder);
    return root;
};

// maximum binary tree
/**
 * @param {number[]} nums
 * @return {TreeNode}
 */
 var constructMaximumBinaryTree = function(nums) {
    if (nums.length == 1) {
        return new TreeNode(nums[0]);
    }
    
    const helper = (arr, leftIndex, rightIndex) => {
        if (leftIndex > rightIndex) {
            return null;
        }

        let maxValue = -1;
        let maxIndex = -1;
        for (let i = leftIndex; i <= rightIndex; i++) {
            if (arr[i] > maxValue) {
                maxValue = arr[i];
                maxIndex = i;
            }
        }
        let root = new TreeNode(maxValue);
        root.left = helper(arr, leftIndex, maxIndex - 1);
        root.right = helper(arr, maxIndex + 1, rightIndex);
        return root;
    }
    
    let root = helper(nums, 0, nums.length - 1);
    return root;
};

// Merge Two Binary Trees
/**
 * @param {TreeNode} root1
 * @param {TreeNode} root2
 * @return {TreeNode}
 */
 var mergeTrees = function(root1, root2) {
    if (root1 == null) return root2;
    if (root2 == null) return root1;
    
    let root = new TreeNode(root1.val + root2.val);
    root.left = mergeTrees(root1.left, root2.left);
    root.right = mergeTrees(root1.right, root2.right);
    return root;
};

// Search in a Binary Search Tree
/**
 * @param {TreeNode} root
 * @param {number} val
 * @return {TreeNode}
 */
 var searchBST = function(root, val) {
    if (root == null || root.val == val)
        return root;
    if (root.val > val) return searchBST(root.left, val);
    if (root.val < val) return searchBST(root.right, val);
    
    return null;
};

// Validate Binary Search Tree
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
 var isValidBST = function(root) {
    let cur = null;
    const inorder = (root) => {
         if (root == null) return true;
    
        let left = inorder(root.left);
        if (!left) return false;

        if (cur != null && cur.val >= root.val) {
            return false;
        }
        cur = root;
        let right = inorder(root.right);
        return right;
    }
    return inorder(root);
};

// Minimum Absolute Difference in BST
/**
 * @param {TreeNode} root
 * @return {number}
 */
 var getMinimumDifference = function(root) {
    let res = Number.MAX_VALUE;
    let pre = null;
    const traverse = cur => {
        if (cur == null) return;
        traverse(cur.left);
        if (pre != null) {
            res = Math.min(res, cur.val - pre.val);
        }
        pre = cur;
        traverse(cur.right);
    }
    traverse(root);
    return res;
};

// Find Mode in Binary Search Tree
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var findMode = function(root) {
    let count = 0, maxCount = 0;
    let pre = null;
    let res = [];
    const helper = cur => {
        if (cur == null) return;
        helper(cur.left);
        if (pre == null || pre.val != cur.val) {
            count = 1;
        } else {
            count++;
        }
        if (count > maxCount) {
            res = [];
            res.push(cur.val);
            maxCount = count;
        } else if (count == maxCount) {
            res.push(cur.val);
        }
        pre = cur;
        helper(cur.right);
    }
    helper(root);
    return res;
};

// Lowest Common Ancestor of a Binary Tree
/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
 var lowestCommonAncestor = function(root, p, q) {
    if (root == p || root == q || root == null) return root;
    let left = lowestCommonAncestor(root.left, p, q);
    let right = lowestCommonAncestor(root.right, p, q);
    if (left != null && right != null) return root;
    if (left == null) return right;
    return left;
};

// Lowest Common Ancestor of a Binary Search Tree
/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
 var lowestCommonAncestor = function(root, p, q) {
    while (root != null) {
        if (root.val > p.val && root.val > q.val) {
            root = root.left;
        } else if (root.val < p.val && root.val < q.val) {
            root = root.right;
        } else {
            return root;
        }
    }
    return null;
};

// Insert into a Binary Search Tree
/**
 * @param {TreeNode} root
 * @param {number} val
 * @return {TreeNode}
 */
 var insertIntoBST = function(root, val) {
    if (root == null) {
        let node = new TreeNode(val);
        return node;
    }
    if (root.val > val) {
        root.left = insertIntoBST(root.left, val);
    } else {
        root.right = insertIntoBST(root.right, val);
    }
    return root;
};

// Delete Node in a BST
/**
 * @param {TreeNode} root
 * @param {number} key
 * @return {TreeNode}
 */
 var deleteNode = function(root, key) {
    if (root == null) return root;
    
    if (root.val > key) {
        root.left = deleteNode(root.left, key);
    } else if (root.val < key) {
        root.right = deleteNode(root.right, key);
    } else {
        if (root.left == null) return root.right;
        if (root.right == null) return root.left;
        let cur = root.right;
        while (cur.left != null) {
            cur = cur.left;
        }
        cur.left = root.left;
        let tmp = root;
        root = root.right;
        delete tmp;
        return root;
    }
    return root;
};
// trim a binary search tree
/**
 * @param {TreeNode} root
 * @param {number} low
 * @param {number} high
 * @return {TreeNode}
 */
 var trimBST = function(root, low, high) {
    if (root == null) return root;
    if (root.val < low) {
        return trimBST(root.right, low, high);
    }
    if (root.val > high) {
        return trimBST(root.left, low, high);
    }
    root.left = trimBST(root.left, low, high);
    root.right = trimBST(root.right, low, high);
    return root;
};

// Convert Sorted Array to BST
/**
 * @param {number[]} nums
 * @return {TreeNode}
 */
 var sortedArrayToBST = function(nums) {
    const traverse = (nums, left, right) => {
        if (left > right) return null;
        let mid = Math.floor(left + (right - left) / 2);
        let root = new TreeNode(nums[mid]);
        root.left = traverse(nums, left, mid - 1);
        root.right = traverse(nums, mid + 1, right);
        return root;
    };
    let res = traverse(nums, 0, nums.length - 1);
    return res;
};

// Convert BST to Greater Tree
/**
 * @param {TreeNode} root
 * @return {TreeNode}
 */
var convertBST = function(root) {
    let pre = 0;
    const traverse = cur => {
        if (cur == null) return;
        traverse(cur.right);
        cur.val += pre;
        pre = cur.val;
        traverse(cur.left);
    };
    traverse(root);
    return root;
};

// Combinations
/**
 * @param {number} n
 * @param {number} k
 * @return {number[][]}
 */
 var combine = function(n, k) {
    let res = [];
    let path = [];
    const backtracking = (n, k, startIndex) => {
        if (path.length == k) {
            res.push([...path]);
            return;
        }
        for (let i = startIndex; i <= n - k + path.length + 1; i++) {
            path.push(i);
            backtracking(n, k, i + 1);
            path.pop();
        }
    };
    backtracking(n, k, 1);
    return res;
};

// Combination Sum II
/**
 * @param {number[]} candidates
 * @param {number} target
 * @return {number[][]}
 */
 var combinationSum2 = function(candidates, target) {
    let res = [];
    let path = [];
    const backtrack = (sum, startIndex) => {
        if (sum == target) {
            res.push([...path]);
            return;
        }
        for (let i = startIndex; i < candidates.length && sum + candidates[i] <= target; i++) {
            if (i > startIndex && candidates[i] == candidates[i- 1]) continue;
            sum += candidates[i];
            path.push(candidates[i]);
            backtrack(sum, i + 1);
            path.pop();
            sum -= candidates[i];
            
        }
    };
    candidates.sort((a, b) => a - b);
    backtrack(0, 0);
    return res;
};

// Combination Sum III
/**
 * @param {number} k
 * @param {number} n
 * @return {number[][]}
 */
 var combinationSum3 = function(k, n) {
    let path = [];
    let res = [];
    const traverse = (k, n, sum, startIndex) => {
        if (path.length == k) {
            if (sum == n) res.push([...path]);
            return;
        }
        for (let i = startIndex; i <= 9 - k + path.length + 1; i++) {
            sum += i;
            path.push(i);
            traverse(k, n, sum, i + 1);
            sum -= i;
            path.pop();
        }
    };
    traverse(k, n, 0, 1);
    return res;
};

// Letter Combinations of a Phone Numbers
/**
 * @param {string} digits
 * @return {string[]}
 */
 var letterCombinations = function(digits) {
    let res = [];
    if (digits == null || digits.length == 0) return res;
    
    let path = "";
    const myArr = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"];
    
    const backtrack = (digits, startIndex) => {
        if (startIndex == digits.length) {
            res.push(path);
            return;
        }
        const tmp = myArr[digits[startIndex] - '0'];
        for (let i = 0; i < tmp.length; i++) {
            path += tmp[i];
            backtrack(digits, startIndex + 1);
            path = path.slice(0, -1);
        }
    };
    backtrack(digits, 0);
    return res;
};

// Palindrome Partitioning
/**
 * @param {string} s
 * @return {string[][]}
 */
 const isPalindrome = (s, start, end) => {
    for (let i = start, j = end; i < j; i++, j--) {
        if (s[i] != s[j]) {
            return false;
        }
    }
    return true;
};
var partition = function(s) {
    let res = [];
    let path = [];
    const backtrack = (s, startIndex) => {
        if (startIndex >= s.length) {
            res.push([...path]);
            return;
        }
        for (let i = startIndex; i < s.length; i++) {
            if (isPalindrome(s, startIndex, i)) {
                path.push(s.substr(startIndex, i - startIndex + 1));
            } else {
                continue;
            }
            backtrack(s, i + 1);
            path.pop();
        }  
    };
    backtrack(s, 0);
    return res;
};

// Restore IP Addresses
/**
 * @param {string} s
 * @return {string[]}
 */
 var restoreIpAddresses = function(s) {
    let res = [];
    let path = [];
    if (s.length > 12) return res;
    
    const backtrack = (startIndex) => {
        const len = path.length;
        if (len > 4) return;
        if (len == 4 && startIndex == s.length) {
            res.push(path.join("."));
            return;
        }
        for (let i = startIndex; i < s.length; i++) {
            const str = s.substr(startIndex, i - startIndex + 1);
            if (str.length > 3 || +str > 255) break;
            if (str.length > 1 && str[0] == '0') break;
            path.push(str);
            backtrack(i + 1);
            path.pop();
        }
    };
    
    backtrack(0);
    return res;
};

// Subsets
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var subsets = function(nums) {
    let res = [];
    let path = [];
    const backtrack = (startIndex) => {
        res.push([...path]);
        if (startIndex >= nums.length) return;
        for (let i = startIndex; i < nums.length; i++) {
            path.push(nums[i]);
            backtrack(i + 1);
            path.pop();
        }
    };
    backtrack(0);
    return res;
};

// Subsets II
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var subsetsWithDup = function(nums) {
    let res = [];
    let path = [];
    nums.sort((a, b) => a - b);
    const backtrack = startIndex => {
        res.push([...path]);
        for (let i = startIndex; i < nums.length; i++) {
            if (i > startIndex && nums[i] == nums[i - 1]) {
                continue;
            }
            path.push(nums[i]);
            backtrack(i + 1);
            path.pop();
        }
    };
    backtrack(0);
    return res;
};

// Increasing Subsequences
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var findSubsequences = function(nums) {
    let res = [];
    let path = [];
    const backtrack = startIndex => {
        if (path.length > 1) {
            res.push([...path]);
        }
        let arr = new Array(201);
        for (let i = startIndex; i < nums.length; i++) {
            if ((path.length > 0 && nums[i] < path[path.length - 1]) || arr[nums[i] + 100] == 1) continue;
            arr[nums[i] + 100] = 1;
            path.push(nums[i]);
            backtrack(i + 1);
            path.pop();
        }
    };
    backtrack(0);
    return res;
};

// Permutations
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var permute = function(nums) {
    let res = [];
    let path = [];
    let used = new Array(nums.length);
    const backtrack = () => {
        if (path.length == nums.length) {
            res.push([...path]);
            return;
        }
        for (let i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            path.push(nums[i]);
            backtrack();
            path.pop();
            used[i] = false;
        }
    };
    backtrack();
    return res;
};

// Permutations II
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var permuteUnique = function(nums) {
    let res = [];
    let path = [];
    nums.sort((a, b) => a - b);
    const backtrack = (used) => {
        if (path.length == nums.length) {
            res.push([...path]);
            return;
        }
        for (let i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i - 1] == nums[i] && !used[i - 1]) continue;
            if (!used[i]) {
                used[i] = true;
                path.push(nums[i]);
                backtrack(used);
                path.pop();
                used[i] = false;
            }
        }
    };
    backtrack([]);
    return res;
};

// Reconstruct Itinerary
/**
 * @param {string[][]} tickets
 * @return {string[]}
 */
 var findItinerary = function(tickets) {
    let res = [];
    let map = new Map();
    for (let t of tickets) {
        if (!map.has(t[0])) {
            map.set(t[0], []);
        }
        map.get(t[0]).push(t[1]);
    }
    for (let m of map.values()) {
        m.sort();
    }
    // console.log(map);
    const backtrack = start => {
        if (res.length == tickets.length + 1) return true;
        if (!map.has(start) || map.get(start).length == 0) return false;
        
        for (let i = 0 ; i < map.get(start).length; i++) {
            let city = map.get(start)[i];
            map.get(start).splice(i, 1);
            res.push(city);
            if (backtrack(city)) return true;
            res.pop();
            map.get(start).splice(i, 0, city);
        }
    };
    res.push("JFK");
    backtrack("JFK");
    return res;
};

// N queens
/**
 * @param {number} n
 * @return {string[][]}
 */
 var solveNQueens = function(n) {
    let res = [];
    let board = new Array(n);
    for (let i = 0; i < n; i++) {
        board[i] = new Array(n);
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            board[i][j] = ".";
        }
    }
    const isValid = (row, col, board) => {
        for (let i = 0 ; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        for (let i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (let i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    };
    
    const arrToList = board => {
        let list = [];
        for (let i = 0; i < n; i++) {
            list.push(board[i].join(""));
        }
        return list;
    };
    
    const backtrack = row => {
        if (row == n) {
            res.push(arrToList(board));
            return;
        }
        for (let col = 0; col < n; col++) {
            if (isValid(row, col, board)) {
                board[row][col] = 'Q';
                backtrack(row + 1);
                board[row][col] = '.';
            }
        }
    };
    
    backtrack(0);
    return res;
};

// Sudoku Solver
/**
 * @param {character[][]} board
 * @return {void} Do not return anything, modify board in-place instead.
 */
 var solveSudoku = function(board) {
    const isValid = (row, col, k, board) => {
        for (let i = 0; i < 9; i++) {
            if (board[row][i] == String.fromCharCode(k + 48)) return false;
        }
        for (let i = 0; i < 9; i++) {
            if (board[i][col] == String.fromCharCode(k + 48)) return false;
        }
        let sRow = Math.floor(row / 3) * 3;
        let sCol = Math.floor(col / 3) * 3;
        
        for (let i = sRow; i < sRow + 3; i++) {
            for (let j = sCol; j < sCol + 3; j++) {
                if (board[i][j] == String.fromCharCode(k + 48)) return false; 
            }
        }
        return true;   
    };
    const helper = board => {
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                if (board[i][j] != '.') continue;
                for (let k = 1; k <= 9; k++) {
                    if (isValid (i, j, k, board)) {
                        board[i][j] = String.fromCharCode(k + 48);
                        if (helper(board)) return true;
                        board[i][j] = '.';
                    }
                }
                return false;
            }
        }
        return true;
    };
    helper(board);
};

// Assign Cookies
/**
 * @param {number[]} g
 * @param {number[]} s
 * @return {number}
 */
 var findContentChildren = function(g, s) {
    g.sort((a, b) => a - b);
    s.sort((a, b) => a - b);
    let res = 0;
    let index = s.length - 1;
    for (let i = g.length - 1; i >= 0; i--) {
        if (index >= 0 && s[index] >= g[i]) {
            res++;
            index--;
        }
    }
    return res;
};

// Wiggle Subsequence
/**
 * @param {number[]} nums
 * @return {number}
 */
 var wiggleMaxLength = function(nums) {
    if (nums.length <= 1) return nums.length;
    let res = 1;
    let curDiff = 0, preDiff = 0;
    for (let i = 0; i < nums.length - 1; i++) {
        curDiff = nums[i + 1] - nums[i];
        if ((preDiff <= 0 && curDiff > 0) || (preDiff >= 0 && curDiff < 0)) {
            res++;
            preDiff = curDiff;
        }
    }
    return res;
};

// Best Time to Buy and Sell Stock II
/**
 * @param {number[]} prices
 * @return {number}
 */
 var maxProfit = function(prices) {
    let res = 0;
    for (let i = 1; i < prices.length; i++) {
        res += Math.max(prices[i] - prices[i - 1], 0);
    }
    return res;
};

// Jump Game
/**
 * @param {number[]} nums
 * @return {boolean}
 */
 var canJump = function(nums) {
    if (nums.length == 1) return true;
    let cover = 0;
    for (let i = 0; i <= cover; i++) {
        cover = Math.max(i + nums[i], cover);
        if (cover >= nums.length - 1) return true;
    }
    return false;
};

// Jump Game II
/**
 * @param {number[]} nums
 * @return {number}
 */
 var jump = function(nums) {
    if (nums.length == 1) return 0;
    let cur = 0, next = 0, res = 0;
    for (let i = 0; i < nums.length - 1; i++) {
        next = Math.max(nums[i] + i, next);
        if (i == cur) {
            cur = next;
            res++;
        }
    }
    return res;
};

// Gas Station
/**
 * @param {number[]} gas
 * @param {number[]} cost
 * @return {number}
 */
 var canCompleteCircuit = function(gas, cost) {
    let start = 0;
    let totalSum = 0;
    let curSum = 0;
    for (let i = 0; i < gas.length; i++) {
        curSum += gas[i] - cost[i];
        totalSum += gas[i] - cost[i];
        if (curSum < 0) {
            curSum = 0;
            start = i + 1;
        }
    }
    if (totalSum < 0) return -1;
    return start;
};

// Candy
/**
 * @param {number[]} ratings
 * @return {number}
 */
 var candy = function(ratings) {
    let candies = new Array(ratings.length).fill(1);
    for (let i = 1; i < candies.length; i++) {
        if (ratings[i] > ratings[i - 1]) {
            candies[i] = candies[i - 1] + 1;
        }
    }
    for (let i = candies.length - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1]) {
            candies[i] = Math.max(candies[i + 1] + 1, candies[i]);
        }
    }
    let res = candies.reduce((a, b) => a + b, 0);
    return res;
};

// Lemonade Change
/**
 * @param {number[]} bills
 * @return {boolean}
 */
 var lemonadeChange = function(bills) {
    let five = 0;
    let ten = 0;
    for (let i = 0; i < bills.length; i++) {
        if (bills[i] == 5) {
            five++;
        } else if (bills[i] == 10) {
            if (five <= 0) return false;
            five--;
            ten++;
        } else {
            if (five > 0 && ten > 0) {
                five--;
                ten--;
            } else if (five >= 3) {
                five -=3;
            } else {
                return false;
            }
        }
    }
    return true;
};

// Queue Reconstruction by Height
/**
 * @param {number[][]} people
 * @return {number[][]}
 */
 var reconstructQueue = function(people) {
    people.sort((a, b) => {
        if (a[0] == b[0]) {
            return a[1] - b[1]; 
        }
        return b[0] - a[0];
    });
    let res = [];
    for (let i = 0; i < people.length; i++) {
        res.splice(people[i][1], 0, people[i]);
    }
    return res;
};

// Minimum Number of Arrows to Burst Balloons
/**
 * @param {number[][]} points
 * @return {number}
 */
 var findMinArrowShots = function(points) {
    points.sort((a, b) => a[0] - b[0]);
    let count = 1
    for (let i = 1; i < points.length; i++) {
        if (points[i][0] > points[i - 1][1]) {
            count++;
        } else {
            points[i][1] = Math.min(points[i - 1][1], points[i][1]);
        }
    }
    return count;
};


// Non-overlapping Intervals
/**
 * @param {number[][]} intervals
 * @return {number}
 */
 var eraseOverlapIntervals = function(intervals) {
    if (intervals.length == 0) return 0;
    intervals.sort((a, b) => a[1] - b[1]);
    let count = 1;
    let end = intervals[0][1];
    for (let i = 0; i < intervals.length; i++) {
        if (end <= intervals[i][0]) {
            count++;
            end = intervals[i][1];
        }
    }
    return intervals.length - count;
};

// Maximize Sum Of Array After K Negations
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
 var largestSumAfterKNegations = function(nums, k) {
    nums.sort((a, b) => Math.abs(b) - Math.abs(a));
    
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] < 0 && k > 0) {
            nums[i] = -nums[i];
            k--;
        }
    }

    if (k % 2 == 1) nums[nums.length - 1] = -nums[nums.length - 1];
    return nums.reduce((a,b) => a + b);
};

// Maximum Subarray
/**
 * @param {number[]} nums
 * @return {number}
 */
 var maxSubArray = function(nums) {
    let res = Number.MIN_SAFE_INTEGER;
    let count = 0;
    for (let i = 0; i < nums.length; i++) {
        count += nums[i];
        if (res < count) res = count;
        if (count <= 0) count = 0;
    }
    return res;
};

// implement queue using stacks
/**
 * Initialize your data structure here.
 */
 var MyQueue = function() {
    this.sIn = [];
    this.sOut = [];
};

// Balanced Binary Tree
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
 var isBalanced = function(root) {
    let res = getHeight(root);
    return res != -1
};
function getHeight(root) {
    if (root == null)
        return 0;
    let leftHeight = getHeight(root.left);
    if (leftHeight == -1)
        return -1;
    let rightHeight = getHeight(root.right);
    if (rightHeight == -1) 
        return -1;
    return Math.abs(leftHeight - rightHeight) > 1 ? -1 : 1 + Math.max(leftHeight, rightHeight);
}

/**
 * Push element x to the back of queue. 
 * @param {number} x
 * @return {void}
 */
MyQueue.prototype.push = function(x) {
    this.sIn.push(x);
};

/**
 * Removes the element from in front of queue and returns that element.
 * @return {number}
 */
MyQueue.prototype.pop = function() {
    if (this.sOut.length == 0) {
        while (this.sIn.length != 0) {
            this.sOut.push(this.sIn.pop());
        }
    }
    return this.sOut.pop();
};

/**
 * Get the front element.
 * @return {number}
 */
MyQueue.prototype.peek = function() {
    let res = this.pop();
    this.sOut.push(res);
    return res;
};

/**
 * Returns whether the queue is empty.
 * @return {boolean}
 */
MyQueue.prototype.empty = function() {
    return this.sIn.length == 0 && this.sOut.length == 0;
};

// implement stack using queues
/**
 * Initialize your data structure here.
 */
 var MyStack = function() {
    this.queue = [];
};

/**
 * Push element x onto stack. 
 * @param {number} x
 * @return {void}
 */
MyStack.prototype.push = function(x) {
    this.queue.push(x);
};

/**
 * Removes the element on top of the stack and returns that element.
 * @return {number}
 */
MyStack.prototype.pop = function() {
    let count = this.queue.length;
    count--;
    while (count > 0) {
        this.queue.push(this.queue.shift());
        count--;
    }
    return this.queue.shift();
};

/**
 * Get the top element.
 * @return {number}
 */
MyStack.prototype.top = function() {
    let res = this.pop();
    this.queue.push(res);
    return res;
};

/**
 * Returns whether the stack is empty.
 * @return {boolean}
 */
MyStack.prototype.empty = function() {
    return this.queue.length == 0;
};

// valid parentheses
/**
 * @param {string} s
 * @return {boolean}
 */
 var isValid = function(s) {
    let stack = [];
    for (let i = 0; i < s.length; i++) {
        if (s[i] == '(') {
            stack.push(')');
        } else if (s[i] == '{') {
            stack.push('}');
        } else if (s[i] == '[') {
            stack.push(']');
        } else if (stack.length == 0 || stack[stack.length - 1] != s[i]) {
            return false;
        } else {
            stack.pop();
        }
    }
    return stack.length == 0
};

// Remove All Adjacent Duplicates In String
/**
 * @param {string} s
 * @return {string}
 */
 var removeDuplicates = function(s) {
    let res = [];
    for (let i = 0; i < s.length; i++) {
        if (res.length == 0 || res[res.length - 1] != s[i]) {
            res.push(s[i]);
        } else {
            res.pop();
        }
    }
    return res.join("");
};

// Evaluate Reverse Polish Notation
/**
 * @param {string[]} tokens
 * @return {number}
 */
 var evalRPN = function(tokens) {
    let stack = [];
    for (let i = 0; i < tokens.length; i++) {
        if (tokens[i] == "+") {
            stack.push(stack.pop() + stack.pop());
        } else if (tokens[i] == "-") {
            stack.push(-stack.pop() + stack.pop());
        } else if (tokens[i] == "*") {
            stack.push(stack.pop() * stack.pop());
        } else if (tokens[i] == "/") {
            let t1 = stack.pop();
            let t2 = stack.pop();
            stack.push(t2 / t1 | 0);
        } else {
            stack.push(parseInt(tokens[i]));
        }
    }
    return stack.pop();
};

// top k frequent elements
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */
 var topKFrequent = function(nums, k) {
    if (nums.length == k) return nums;
    const map = new Map();
    for(let n of nums) map.set(n, (map.get(n) || 0) + 1);
    const keys = [...map.keys()], finalIdx = keys.length - k;
    let start = 0, end = keys.length-1;
    
    while(start <= end) {
        const pivot = Math.floor(Math.random() * (end - start + 1)) + start;
        const pivotIdx = pivotHelper(pivot, start, end);
        
        if(pivotIdx === finalIdx) return keys.slice(finalIdx);
        if(pivotIdx < finalIdx) start = pivotIdx + 1;
        else end = pivotIdx - 1;
    }
    
    function pivotHelper(pivot, start, end) {
        // move pivot away to the end
        swap(pivot, end);
        let swapIdx = start;
        
        for(let i = start; i < end; i++) {
            if(map.get(keys[i]) < map.get(keys[end])) {
                swap(swapIdx, i); swapIdx++;
            }
        }
        swap(swapIdx, end);
        return swapIdx;
    }
    
    function swap(i, j) {
        [keys[i], keys[j]] = [keys[j], keys[i]];
    }
};

// Sliding Window Maximum
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */
 var maxSlidingWindow = function(nums, k) {
    let indexArr = [];
    let res = [];
    for (let i = 0; i < nums.length; i++) {
        while (indexArr.length && nums[i] >= nums[indexArr[indexArr.length - 1]]) {
            indexArr.pop();
        }
        indexArr.push(i);
        while (indexArr[0] <= i - k) {
            indexArr.shift();
        }
        if (i >= k - 1) res.push(nums[indexArr[0]]); 
    }
    return res;
};

// Design Linked List
class ListNode {
    constructor(val, next) {
        this.val = val;
        this.next = next;
    }
}
/**
 * Initialize your data structure here.
 */
var MyLinkedList = function () {
    this._head = new ListNode(0, null);
    this._size = 0;
};

/**
 * Get the value of the index-th node in the linked list. If the index is invalid, return -1. 
 * @param {number} index
 * @return {number}
 */
MyLinkedList.prototype.get = function (index) {
    if (index >= this.size || index < 0) return -1;
    let cur = this._head;
    for (let i = 0; i <= index; i++) {
        cur = cur.next;
    }
    return cur.val;
};

/**
 * Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. 
 * @param {number} val
 * @return {void}
 */
MyLinkedList.prototype.addAtHead = function (val) {
    this.addAtIndex(0, val);
};

/**
 * Append a node of value val to the last element of the linked list. 
 * @param {number} val
 * @return {void}
 */
MyLinkedList.prototype.addAtTail = function (val) {
    this.addAtIndex(this._size, val);
};

/**
 * Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. 
 * @param {number} index 
 * @param {number} val
 * @return {void}
 */
MyLinkedList.prototype.addAtIndex = function (index, val) {
    if (index > this._size) return;
    if (index < 0)
        index = 0;
    this._size++;
    let pre = this._head;
    for (let i = 0; i < index; i++) {
        pre = pre.next;
    }
    let cur = new ListNode(val, pre.next);
    pre.next = cur;
};

/**
 * Delete the index-th node in the linked list, if the index is valid. 
 * @param {number} index
 * @return {void}
 */
MyLinkedList.prototype.deleteAtIndex = function (index) {
    if (index >= this._size || index < 0) return;
    this._size--;
    let pre = this._head;
    for (let i = 0; i < index; i++) {
        pre = pre.next;
    }
    pre.next = pre.next.next;
};
