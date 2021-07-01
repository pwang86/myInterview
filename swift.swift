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

    // Remove Linked List Elements
    /**
    * Definition for singly-linked list.
    * public class ListNode {
    *     public var val: Int
    *     public var next: ListNode?
    *     public init() { self.val = 0; self.next = nil; }
    *     public init(_ val: Int) { self.val = val; self.next = nil; }
    *     public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
    * }
    */
     func removeElements(_ head: ListNode?, _ val: Int) -> ListNode? {
        if head == nil {
            return head
        }
        
        let dummy = ListNode(0)
        dummy.next = head
        var pre: ListNode? = dummy
        var cur: ListNode? = head
        
        while cur != nil {
            if cur?.val == val {
                pre.next = cur?.next
            } else {
                pre = cur!
            }
            cur = cur?.next
        }
        
        return dummy.next
    }

    // reverse linked list
    func reverseList(_ head: ListNode?) -> ListNode? {
        var pre: ListNode? = nil
        var tmp: ListNode? = nil
        var cur = head
        while cur != nil {
            tmp = cur?.next
            cur?.next = pre
            pre = cur
            cur = tmp
        }
        return pre
    }

    // swap nodes in pairs
    func swapPairs(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }
        
        let dummy = ListNode(0, head)
        var pre = dummy
        var cur = head
        while cur != nil && cur?.next != nil {
            var tmp = cur?.next?.next
            pre.next = cur?.next
            cur?.next?.next = cur
            cur?.next = tmp
            pre = cur!
            cur = cur?.next
        }
        return dummy.next
    }

    // remove nth node from end of list
    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        var dummy = ListNode(0, head)
        var slow: ListNode? = dummy
        var fast: ListNode? = dummy
        for _ in 1...n {
            fast = fast?.next
        }
        fast = fast?.next
        while fast != nil {
            fast = fast?.next
            slow = slow?.next
        }
        slow?.next = slow?.next?.next
        return dummy.next
    }

    // intersection of two linked lists
    func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
        var curA: ListNode? = headA
        var curB: ListNode? = headB
        var lenA = 0
        var lenB = 0
        while curA != nil {
            lenA += 1
            curA = curA?.next
        }
        while curB != nil {
            lenB += 1
            curB = curB?.next
        }
        curA = headA
        curB = headB
        
        if lenB > lenA {
            swap(&lenA, &lenB)
            swap(&curA, &curB)
        }
        
        var gap = lenA - lenB
        while gap > 0 {
            gap -= 1
            curA = curA?.next
        }
        while curA != nil {
            // compare two reference in swift using === or !==
            if curA === curB {
                return curA
            }
            curA = curA?.next
            curB = curB?.next
        }
        return nil
    }

    // linked list cycle ii
    func detectCycle(_ head: ListNode?) -> ListNode? {
        var fast: ListNode? = head
        var slow: ListNode? = head
        while fast != nil && fast?.next != nil {
            fast = fast?.next?.next
            slow = slow?.next
            if fast === slow {
                var index1: ListNode? = fast
                var index2: ListNode? = head
                while index1 !== index2 {
                    index1 = index1?.next
                    index2 = index2?.next
                }
                return index2
            }
        }
        return nil
    }

    // valid anagram
    func isAnagram(_ s: String, _ t: String) -> Bool {
        if s.count != t.count {
            return false
        }
        var record = [Int](repeating: 0, count: 26)
        for i in s {
            var diff = Int(i.asciiValue! - Character("a").asciiValue!)
            record[diff] += 1
        }
        for i in t {
            var diff =  Int(i.asciiValue! - Character("a").asciiValue!)
            record[diff] -= 1
        }
        for i in 0..<26 {
            if record[i] != 0 {
                return false
            }
        }
        return true
    }

    // intersection of two arrays
    func intersection(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        if nums1 == nil || nums1.count == 0 || nums2 == nil || nums2.count == 0 {
            return []
        }
        var set1: Set<Int> = Set(nums1)
        var res = Set<Int>()
        for num in nums2 {
            if set1.contains(num) {
                res.insert(num)
            }
        }
        return Array(res)
    }

    // happy number
    func isHappy(_ n: Int) -> Bool {
        var mySet = Set<Int>()
        var res = n
        while res != 1 && !mySet.contains(res) {
            mySet.insert(res)
            res = getSum(res)
        }
        return res == 1
    }
    func getSum(_ n: Int) -> Int {
        var sum: Int = 0
        var a = n
        while a > 0 {
            var tmp = a % 10
            sum += tmp * tmp
            a /= 10
        }
        return sum
    }
}

// Deisgn Linked List
class ListNode {
    var val: Int
    var next: ListNode?
    init(_ val: Int) {
        self.val = val
    }    
}

class MyLinkedList {
    var head: ListNode?
    var size = 0
    
    /** Initialize your data structure here. */
    init() {
        self.head = ListNode(0)
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    func get(_ index: Int) -> Int {
        if index >= self.size || index < 0 {
            return -1
        }
        var cur = self.head
        for i in 0...index {
            cur = cur?.next
        }
        return cur!.val
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    func addAtHead(_ val: Int) {
        self.addAtIndex(0, val)
    }
    
    /** Append a node of value val to the last element of the linked list. */
    func addAtTail(_ val: Int) {
        self.addAtIndex(self.size, val)
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    func addAtIndex(_ index: Int, _ val: Int) {
        var tmp = index
        if index < 0 {
            tmp = 0
        }
        if index > self.size {
            return
        }
        self.size += 1
        var pre = self.head
        for i in 0..<tmp {
            pre = pre?.next
        }
        var cur = ListNode(val)
        cur.next = pre?.next
        pre?.next = cur
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    func deleteAtIndex(_ index: Int) {
        if index < 0 || index >= self.size {
            return
        }
        self.size -= 1
        var pre = self.head
        for i in 0..<index {
            pre = pre?.next
        }
        pre?.next = pre?.next?.next
    }
}