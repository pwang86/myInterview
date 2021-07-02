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
