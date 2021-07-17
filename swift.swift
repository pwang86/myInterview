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

    //two sum
     func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
        var dic: [Int: Int] = [:]
        for (index, value) in nums.enumerated() {
            var tmp = target - value
            if dic[tmp] != nil {
                return [index, dic[tmp]!]
            }
            dic[value] = index
        }
        return []
    }

    // 4 sum II
    func fourSumCount(_ nums1: [Int], _ nums2: [Int], _ nums3: [Int], _ nums4: [Int]) -> Int {
        var myMap: [Int : Int] = [:]
        var res = 0
        
        for i in nums1 {
            for j in nums2 {
                if let value = myMap[i + j] {
                    myMap[i + j] = value + 1
                } else {
                    myMap[i + j] = 1
                }
            }
        } 
        
        for i in nums3 {
            for j in nums4 {
                if let value = myMap[0 - i - j] {
                    res += value
                }
            }
        }   
        return res
    }

    // ransome note
    func canConstruct(_ ransomNote: String, _ magazine: String) -> Bool {
        var record = Array(repeating: 0, count: 26)
        var tmp = 0
        
        for i in magazine {
            tmp = Int(i.asciiValue! - Character("a").asciiValue!)
            record[tmp] += 1
        }
        
        for i in ransomNote {
            tmp = Int(i.asciiValue! - Character("a").asciiValue!)
            if record[tmp] > 0 {
                record[tmp] -= 1
            } else {
                return false
            }
        }
        return true
    }

    // 3 sum
    func threeSum(_ nums: [Int]) -> [[Int]] {
        var res:[[Int]] = []
        var copy = nums.sorted()
        
        for i in 0..<copy.count {
            if copy[i] > 0 {
                return res
            }
            if i > 0 && copy[i - 1] == copy[i] {
                continue
            }
            
            var left = i + 1
            var right = copy.count - 1
            while right > left {
                let tmp = copy[i] + copy[left] + copy[right]
                if tmp > 0 {
                    right -= 1
                } else if tmp < 0 {
                    left += 1
                } else {
                    res.append([copy[i], copy[left], copy[right]])
                    while right > left && copy[right] == copy[right - 1] {
                        right -= 1
                    }
                    while right > left && copy[left] == copy[left + 1] {
                        left += 1
                    }
                    right -= 1
                    left += 1
                }
            }
        }
        
        return res
    }

    // 4 sum
     func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
        var res:[[Int]] = []
        var copy = nums.sorted()
        
        for i in 0..<copy.count {
            if i > 0 && copy[i - 1] == copy[i] {
                continue
            }
            for j in i+1..<copy.count {
                if j > i + 1 && copy[j - 1] == copy[j] {
                    continue
                }
                var left = j + 1
                var right = copy.count - 1
                while right > left {
                    let sum = copy[i] + copy[j] + copy[left] + copy[right]
                    if sum > target {
                        right -= 1
                    } else if sum < target {
                        left += 1
                    } else {
                        res.append([copy[i], copy[j], copy[left], copy[right]])
                        while right > left && copy[right - 1] == copy[right] {
                            right -= 1
                        }
                        while right > left && copy[left + 1] == copy[left] {
                            left += 1 
                        }
                        right -= 1
                        left += 1
                    }
                }
            }
        }
        return res
    }

    // reverse string
    func reverseString(_ s: inout [Character]) {
        var l = 0, r = s.count  - 1
        while l < r {
            s.swapAt(l, r)
            l += 1
            r -= 1
        }
    }

    // reverse string II
     func reverseStr(_ s: String, _ k: Int) -> String {
        var tmp = Array(s)
        for i in stride(from: 0, to: tmp.count, by: 2 * k) {
            var start = i
            let a = tmp.count - 1
            let b = i + k - 1
            var end = a >= b ? b : a
            while start < end {
                tmp.swapAt(start, end)
                start += 1
                end -= 1
            }
         }
        return String(tmp)
    }

    // implement strStr()
     func strStr(_ haystack: String, _ needle: String) -> Int {
        if needle.count == 0 {
            return 0
        }
        
        let arrHay = Array(haystack)
        let arrNeedle = Array(needle)
        
        var next = Array(repeating: 0, count: arrNeedle.count)
        var j = -1
        next[0] = j
        for i in 1..<arrNeedle.count {
            while j >= 0 && arrNeedle[i] != arrNeedle[j + 1] {
                j = next[j]
            }
            if arrNeedle[i] == arrNeedle[j + 1] {
                j += 1
            }
            next[i] = j
        }
        
        j = -1
        for i in arrHay.indices {
            while j >= 0 && arrHay[i] != arrNeedle[j + 1] {
                j = next[j]
            }
            if arrHay[i] == arrNeedle[j + 1] {
                j += 1
            }
            if j == arrNeedle.count - 1 {
                return (i - arrNeedle.count + 1) 
            }
        }
        return -1
    }

    // Reverse Words in a string
    func reverseWords(_ s: String) -> String {
        var res:[Character] = removeSpace(s)
        res = reverseString(res, 0, res.count - 1)
        res = reverseEachWord(res)
        return String(res)
    }
    func removeSpace(_ s: String) -> [Character] {
        var l = 0, r = s.count - 1
        var tmp = Array(s)
        while l <= r && tmp[l] == " " {
            l += 1
        }
        while l <= r && tmp[r] == " " {
            r -= 1
        }
        var res = [Character]()
        while l <= r {
            if tmp[l] != " " || res.last != " " {
                res.append(tmp[l])
            }
            l += 1
        }
        return res
    }
    func reverseString(_ arr: [Character], _ start: Int, _ end: Int) -> [Character] {
        var l = start, r = end
        var res = arr
        while l < r {
            res.swapAt(l, r)
            l += 1
            r -= 1
        }
        return res
    }
    func reverseEachWord(_ arr: [Character]) -> [Character] {
        var start = 0, end = 0
        var res = arr
        while start < res.count {
            while end < res.count && res[end] != " " {
                end += 1
            }
            res = reverseString(res, start, end - 1)
            start = end + 1
            end += 1
        }
        return res
    }

    // Repeated Substring Pattern
    func repeatedSubstringPattern(_ s: String) -> Bool {
        if s.count == 0 {
            return false
        }
        
        let len = s.count
        var next:[Int] = getNext(s)
        
        if next[len - 1] != -1, len % (len - (next[len - 1] + 1)) == 0 {
            return true
        }
        return false
    }
    func getNext(_ s: String) -> [Int]{
        var next = Array(repeating: 0, count: s.count)
        let tmp = Array(s)
        var j = -1
        next[0] = j
        for i in 1..<tmp.count {
            while j >= 0 && tmp[i] != tmp[j + 1] {
                j = next[j]
            }
            if tmp[i] == tmp[j + 1] {
                j += 1
            }
            next[i] = j
        }
        return next;
    }
    
    // valid parentheses
    func isValid(_ s: String) -> Bool {
        var stack: [Character] = []
        let arr = Array(s)
        for i in 0..<arr.count {
            if arr[i] == "(" {
                stack.append(")")
            } else if arr[i] == "{" {
                stack.append("}")
            } else if arr[i] == "[" {
                stack.append("]")
            } else if stack.count == 0 || stack[stack.count - 1] != arr[i] {
                return false
            } else {
                stack.removeLast()
            }
        }
        return stack.count == 0
    }

    // Remove All Adjacent Duplicates In String
     func removeDuplicates(_ s: String) -> String {
        var res: [Character] = []
        let tmp = Array(s)
        for i in 0..<s.count {
            if res.count == 0 || res[res.count - 1] != tmp[i] {
                res.append(tmp[i])
            } else {
                res.removeLast()
            }
        }
        return String(res)
    }

    // Evaluate Reverse Polish Notation
    func evalRPN(_ tokens: [String]) -> Int {
        var stack:[Int] = []
        for i in 0..<tokens.count {
            if tokens[i] == "+" {
                stack.append(stack.removeLast() + stack.removeLast())
            } else if tokens[i] == "-" {
                stack.append(-stack.removeLast() + stack.removeLast())
            } else if tokens[i] == "*" {
                stack.append(stack.removeLast() * stack.removeLast())
            } else if tokens[i] == "/" {
                let t1 = stack.removeLast()
                let t2 = stack.removeLast()
                stack.append(t2 / t1)
            } else {
                stack.append(Int(tokens[i])!)
            }
        }
        return stack.removeLast()
    }
}

// implement queue using stacks
class MyQueue {
    var sIn: [Int] = []
    var sOut: [Int] = []
    /** Initialize your data structure here. */
    init() {

    }
    
    /** Push element x to the back of queue. */
    func push(_ x: Int) {
        sIn.append(x)
    }
    
    /** Removes the element from in front of queue and returns that element. */
    func pop() -> Int {
        if sOut.isEmpty {
            while !sIn.isEmpty {
                sOut.append(sIn.removeLast())
            }
        }
        return sOut.removeLast()
    }
    
    /** Get the front element. */
    func peek() -> Int {
        let res = pop()
        sOut.append(res)
        return res
    }
    
    /** Returns whether the queue is empty. */
    func empty() -> Bool {
        return sIn.isEmpty && sOut.isEmpty
    }
}

// implement stack using queues
class MyStack {
    var queue:[Int] = []
    /** Initialize your data structure here. */
    init() {
    }
    
    /** Push element x onto stack. */
    func push(_ x: Int) {
        queue.append(x)
    }
    
    /** Removes the element on top of the stack and returns that element. */
    func pop() -> Int {
        var size = queue.count
        size -= 1
        while size > 0 {
            queue.append(queue.removeLast())
            size -= 1
        }
        return queue.removeLast()
    }
    
    /** Get the top element. */
    func top() -> Int {
        let res = pop()
        queue.append(res)
        return res
    }
    
    /** Returns whether the stack is empty. */
    func empty() -> Bool {
        return queue.count == 0
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