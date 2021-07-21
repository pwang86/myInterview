from typing import List
import heapq

class Solution:
    # spiral matrix II 
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = [[0] * n for i in range(n)]
        loop = n // 2
        startX = startY = 0
        count = 1
        offset = 1
        mid = n // 2
        while loop > 0:
            i = startX
            j = startY
            while j < startY + n - offset:
                res[i][j] = count
                count += 1
                j += 1
            while i < startX + n - offset:
                res[i][j] = count
                count += 1
                i += 1
            while j > startY:
                res[i][j] = count
                count += 1
                j -= 1
            while i > startX:
                res[i][j] = count
                count += 1
                i -= 1
            loop -= 1
            offset += 2
            startX += 1
            startY += 1
        
        if n % 2 == 1:
            res[mid][mid] = count
        
        return res

    # Remove Linked List Elements

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        if head == None:
            return head
        
        dummy = ListNode(next=head)
        pre = dummy
        cur = head
        
        while cur != None:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
        return dummy.next
    
    # reverse linked list
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        tmp = None
        cur = head
        while cur != None:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre

    # swap nodes in pairs
    def swapPairs(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        
        dummy = ListNode(0, head)
        pre = dummy
        cur = head
        while cur != None and cur.next != None:
            tmp = cur.next.next
            pre.next = cur.next
            cur.next.next = cur
            cur.next = tmp
            pre = cur
            cur = cur.next
        
        return dummy.next

    # remove nth node from end of list
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        slow = dummy
        fast = dummy
        while n > 0:
            fast = fast.next
            n -= 1
        fast = fast.next
        while fast != None:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
    
    # intersection of two linked lists
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lenA, lenB = 0, 0
        curA, curB = headA, headB
        while curA != None:
            lenA += 1
            curA = curA.next
        while curB != None:
            lenB += 1
            curB = curB.next
        curA = headA
        curB = headB
        
        if lenB > lenA:
            lenA, lenB = lenB, lenA
            curA, curB = curB, curA
        
        gap = lenA - lenB
        while gap > 0:
            gap -= 1
            curA = curA.next
        while curA != None:
            if curA == curB:
                return curA
            curA = curA.next
            curB = curB.next
        return None

    # Linked list cycle II
    def detectCycle(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while fast and fast.next: 
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                index1 = fast
                index2 = head
                while index1 != index2:
                    index1 = index1.next
                    index2 = index2.next
                return index2 
        return None

    # valid anagram
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        record = [0] * 26
        for i in s:
            record[ord(i) - ord('a')] += 1
        for i in t:
            record[ord(i) - ord('a')] -= 1
            
        for i in range(26):
            if record[i] != 0:
                return False
        return True
    
    # intersection of two arrays
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = set()
        set1 = set(nums1)
        for n in nums2:
            if n in set1:
                res.add(n)
        return list(res)
    
    # happy number
    def isHappy(self, n: int) -> bool:
        mySet = set()
        while 1:
            n = self.getSum(n)
            if n == 1:
                return True
            if n in mySet:
                return False
            else:
                mySet.add(n)

    def getSum(self, n: int) -> int:
        sum = 0
        while n > 0:
            tmp = n % 10
            sum += tmp * tmp
            n //= 10
        return sum
    
    # two sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = [None] * 2
        dic = {}
        for i, num in enumerate(nums):
            tmp = target - num
            if dic.get(tmp) != None:
                return [i, dic.get(tmp)]
            dic[num] = i
        return res
    
    # 4sum II
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        myMap = dict()
        for ni in nums1:
            for j in nums2:
                if i + j in myMap:
                    myMap[i + j] += 1
                else:
                    myMap[i + j] = 1
        
        # if the -(a+b) exists in nums3 and nums4, we shall add the count
        count = 0
        for i in nums3:
            for j in nums4:
                if 0 - i - j in myMap:
                    count += myMap[0 - i - j]
        return count
    
    # ransome Note
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        record = [0] * 26
        tmp = 0
        
        for i in magazine:
            record[ord(i) - ord('a')] += 1
                                
        for i in ransomNote:
            tmp = ord(i) - ord('a')
            if record[tmp] > 0:
                record[tmp] -= 1
            else:
                return False
        return True
    
    # 3 sum
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        
        for i in range(len(nums)):
            if nums[i] > 0:
                return res
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            
            left = i + 1
            right = len(nums) - 1
            while right > left:
                tmp = nums[i] + nums[left] + nums[right]
                if tmp > 0:
                    right -= 1
                elif tmp < 0:
                    left += 1
                else:
                    res.append([nums[i], nums[left], nums[right]])
                    while right > left and nums[right] == nums[right - 1]:
                        right -= 1
                    while right > left and nums[left] == nums[left + 1]: 
                        left += 1
                    right -= 1
                    left += 1
                
        return res
    
    # 4 sum
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []
        nums.sort()
        
        for i in range(len(nums)):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            for j in range(i + 1, len(nums)):
                if j > i + 1 and nums[j - 1] == nums[j]:
                    continue
                left = j + 1
                right = len(nums) - 1
                while right > left:
                    sum = nums[i] + nums[j] + nums[left] + nums[right]
                
                    if sum > target:
                        right -= 1
                    elif sum < target:
                        left += 1
                    else:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while right > left and nums[right - 1] == nums[right]:
                            right -= 1
                        while right > left and nums[left + 1] == nums[left]:
                            left += 1
                        left += 1
                        right -= 1
        return res
    
    # reverse string
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        l, r = 0, len(s) - 1
        while l < r:
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1
    
    # reverse string II
    def reverseStr(self, s: str, k: int) -> str:
        tmp = list(s)
        for i in range(0, len(s), 2 * k):
            start = i
            a = start + k - 1
            b = len(s) - 1
            end = b if a >= b else a
            while start < end:
                tmp[start], tmp[end] = tmp[end], tmp[start]
                start += 1
                end -= 1
        return ''.join(tmp)
    
    # implement strStr
    def strStr(self, haystack: str, needle: str) -> int:
        if len(needle) == 0:
            return 0
        next = [0] * len(needle)
        self.getNext(next, needle)
        j = -1
        for i in range(len(haystack)):
            while j >= 0 and haystack[i] != needle[j + 1]:
                j = next[j]
            if haystack[i] == needle[j + 1]:
                j += 1
            if j == len(needle) - 1:
                return i - len(needle) + 1
        return -1
        
    def getNext(slef, next: List[int], s: str) -> None:
        j = -1
        next[0] = j
        for i in range(1, len(s)):
            while j >= 0 and s[i] != s[j + 1]:
                j = next[j]
            if s[i] == s[j + 1]:
                j += 1
            next[i] = j
    
    # reverse words in a string
    def reverseWords(self, s: str) -> str:
        res = self.removeSpace(s)
        self.reverseString(res, 0 , len(res) - 1)
        self.reverseEachWord(res)
        return ''.join(res)
    def removeSpace(self, s: str) -> List[int]:
        l, r = 0, len(s) - 1
        while l <= r and s[l] == ' ':
            l += 1
        while l <= r and s[r] == ' ':
            r -= 1
        res = []
        while l <= r:
            if s[l] != ' ' or res[- 1] != ' ':
                res.append(s[l])
            l += 1
        return res        
    def reverseString(self, arr: List[int], start: int, end: int) -> None:
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    def reverseEachWord(self, arr: List[int]) -> None:
        l = r = 0
        while l < len(arr):
            while r < len(arr) and arr[r] != ' ':
                r += 1
            self.reverseString(arr, l , r - 1)
            l = r + 1
            r += 1
    
    # Repeated Substring Pattern
    def repeatedSubstringPattern(self, s: str) -> bool:
        if len(s) == 0:
            return False
        
        length = len(s)
        next = [0] * length
        self.getNext(next, s)
        
        if next[- 1] != -1 and length % (length - (next[length - 1] + 1)) == 0:
            return True
        
        return False
    
    def getNext(self, next: List[int], s: str):
        j = -1
        next[0] = j
        for i in range(1, len(s)):
            while j >= 0 and s[i] != s[j + 1]:
                j = next[j]
            if s[i] == s[j + 1]:
                j += 1
            next[i] = j
    
    # valid parentheses
    def isValid(self, s: str) -> bool:
        stack = list()
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(')')
            elif s[i] == '{':
                stack.append('}')
            elif s[i] == '[':
                stack.append(']')
            elif len(stack) == 0 or stack[-1] != s[i]:
                return False
            else:
                stack.pop()
        return len(stack) == 0
    
    # Remove All Adjacent Duplicates In String
    def removeDuplicates(self, s: str) -> str:
        res = list()
        for i in range(len(s)):
            if len(res) == 0 or res[-1] != s[i]:
                res.append(s[i])
            else:
                res.pop()
        return "".join(res)
    
    # Evaluate Reverse Polish Notation
    def evalRPN(self, tokens: List[str]) -> int:
        stack = list()
        for i in range(len(tokens)):
            if tokens[i] not in ["+", "-", "*", "/"]:
                stack.append(tokens[i])
            else:
                t1 = stack.pop()
                t2 = stack.pop()
                res = eval(t2+tokens[i]+t1)
                stack.append(str(int(res)))
        return stack[-1]
    
    # top k frequent elements
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if len(nums) == k:
            return nums
        
        map = {}
        for i in range(len(nums)):
            map[nums[i]] = map.get(nums[i], 0) + 1
        
        que = []
        for key, val in map.items():
            heapq.heappush(que, (val, key))
            if len(que) > k:
                heapq.heappop(que)
        
        res = [0] * k
        for i in range(k - 1, -1, -1):
            res[i] = heapq.heappop(que)[1]
        return res

    # Binary Tree Preorder Traversal
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = list()
        if root == None:
            return res
        self.preOrder(root, res)
        # self.helper(root, res)
        return res
    def preOrder(self, root: TreeNode, res: List[int]) -> None:
        if root == None:
            return
        res.append(root.val)
        self.preOrder(root.left, res)
        self.preOrder(root.right, res) 
    # PostOrder Traversal
    def helper(self, root: TreeNode, res: List[int]) -> None:
        if root == None:
            return
        else:
            self.helper(root.left, res)
            self.helper(root.right, res)
            res.append(root.val)
    
# Sliding Window Maximum
class MyQueue:
    def __init__(self):
        self.myList = list()
    def push(self, val: int) -> None:
        while self.myList and val > self.myList[-1]:
            self.myList.pop()
        self.myList.append(val)
    def pop(self, val: int) -> None:
        if self.myList and val == self.myList[0]:
            self.myList.pop(0)
    def peek(self) -> int:
        return self.myList[0]
        
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if len(nums) == 1:
            return nums
        mq = MyQueue()
        res = list()
        for i in range(k):
            mq.push(nums[i])
        res.append(mq.peek())
        for i in range(k, len(nums)):
            mq.pop(nums[i - k])
            mq.push(nums[i])
            res.append(mq.peek())
        return res

#implement queue using stacks
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.sIn = list()
        self.sOut = list()

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.sIn.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if len(self.sOut) == 0:
            while len(self.sIn) != 0:
                self.sOut.append(self.sIn.pop())
        return self.sOut.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        res = self.pop()
        self.sOut.append(res)
        return res

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.sIn) == 0 and len(self.sOut) == 0

# implement stack using queues
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = list()

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.queue.append(x)

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        count = len(self.queue)
        count -= 1
        while count > 0:
            self.queue.append(self.queue.pop())
            count -= 1
        return self.queue.pop()

    def top(self) -> int:
        """
        Get the top element.
        """
        res = self.pop()
        self.queue.append(res)
        return res

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.queue) == 0



# Deisgn Linked List
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        
class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._head = ListNode(0)
        self._size = 0

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index < 0 or index >= self._size:
            return -1
        cur = self._head
        for i in range(index + 1):
            cur = cur.next
        return cur.val
    
    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self._size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index < 0:
            index = 0
        if index > self._size:
            return
        
        self._size += 1
        pre = self._head
        for i in range(index):
            pre = pre.next
        cur = ListNode(val)
        cur.next = pre.next
        pre.next = cur
        

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < 0 or index >= self._size:
            return
        self._size -= 1
        pre = self._head
        for i in range(index):
            pre = pre.next
        pre.next = pre.next.next