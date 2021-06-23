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