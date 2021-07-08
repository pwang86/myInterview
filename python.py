from typing import List

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