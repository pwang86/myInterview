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

// implement queue using stacks
/**
 * Initialize your data structure here.
 */
 var MyQueue = function() {
    this.sIn = [];
    this.sOut = [];
};

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
