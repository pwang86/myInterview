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