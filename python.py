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
        # self.preOrder(root, res)

        # iterative way
        stack = list()
        stack.append(root)
        while len(stack) != 0:
            tmp = stack.pop()
            res.append(tmp.val)
            if tmp.right != None:
                stack.append(tmp.right)
            if tmp.left != None:
                stack.append(tmp.left)
                
        return res
    def preOrder(self, root: TreeNode, res: List[int]) -> None:
        if root == None:
            return
        res.append(root.val)
        self.preOrder(root.left, res)
        self.preOrder(root.right, res) 

    # PostOrder Traversal
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = list()
        if root == None:
            return res
        # self.helper(root, res)
        stack = list()
        stack.append(root)
        while stack:
            tmp = stack.pop()
            res.append(tmp.val)
            if tmp.left:
                stack.append(tmp.left)
            if tmp.right:
                stack.append(tmp.right)
        return res[::-1]
    def helper(self, root: TreeNode, res: List[int]) -> None:
        if root == None:
            return
        else:
            self.helper(root.left, res)
            self.helper(root.right, res)
            res.append(root.val)

    # InOrder Traversal
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = list()
        if root == None:
            return res
        
        # self.inOrder(root, res)
        stack = list()
        cur = root
        while cur != None or len(stack) != 0:
            if cur != None:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res
    def inOrder(self, root: TreeNode, res: List[int]) -> None:
        if root == None:
            return
        self.inOrder(root.left, res)
        res.append(root.val)
        self.inOrder(root.right, res)

    # Binary Tree Level Order Traversal
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if root == None:
            return res
        
        queue = []
        queue.append(root)
        while (len(queue) != 0):
            size = len(queue)
            tmp = list()
            for i in range(size):
                cur = queue.pop(0)
                tmp.append(cur.val)
                if cur.left != None:
                    queue.append(cur.left)
                if cur.right != None:
                    queue.append(cur.right)
            res.append(tmp)
        
        return res

    # Binary Tree Level Order Traversal II.
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        res = []
        if root == None:
            return res
        self.dfs(root, 0, res)
        return res[::-1]
    
    def dfs(self, root: TreeNode, level: int, res: List[List[int]]) -> None:
            if level >= len(res):
                res.append([])
            res[level].append(root.val)
            if root.left != None:
                self.dfs(root.left, level + 1, res)
            if root.right != None:
                self.dfs(root.right, level + 1, res)
    
    # Binary Tree Right Side View
    def rightSideView(self, root: TreeNode) -> List[int]:
        res = list()
        if root == None:
            return res
        queue = list()
        queue.append(root)
        while len(queue) > 0:
            size = len(queue)
            for i in range(size):
                cur = queue.pop(0)
                if cur.left != None:
                    queue.append(cur.left)
                if cur.right != None:
                    queue.append(cur.right)
                if i == size - 1:
                    res.append(cur.val)
        
        return res
    
    # Average of Levels in Binary Tree
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        res = list()
        if root == None:
            return res
        
        queue = list()
        queue.append(root)
        while len(queue) > 0:
            size = len(queue)
            sum = 0
            for i in range(size):
                cur = queue.pop(0)
                sum += cur.val
                if cur.left != None:
                    queue.append(cur.left)
                if cur.right != None:
                    queue.append(cur.right)
            res.append(sum / size)
        return res

    # Populating Next Right Pointers in Each Node
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        
        queue = list()
        queue.append(root)
        cur = None
        pre = None
        
        while len(queue) > 0:
            size = len(queue)
            for i in range(size):
                if i == 0:
                    pre = queue.pop(0)
                    cur = pre
                else:
                    cur = queue.pop(0)
                    pre.next = cur
                    pre = pre.next
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            pre.next = None
        return root
    
    # Invert Binary Tree
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root == None:
            return root
        
        queue = list()
        queue.append(root)
        while len(queue) > 0:
            size = len(queue)
            for i in range(size):
                cur = queue.pop(0)
                cur.left, cur.right = cur.right,cur.left
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
        
        return root
    # recursive way
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root == None:
            return root
        
        (root.left, root.right) = (root.right, root.left)
        self.invertTree(root.left)
        self.invertTree(root.right)
        
        return root
    
    # Symmetric Tree
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        return self.helper(root.left, root.right)
    
    def helper(self, l: TreeNode, r: TreeNode) -> bool:
        if not l and not r:
            return True
        elif l == None and r != None:
            return False
        elif l != None and r == None:
            return False
        elif l.val != r.val:
            return False
        else:
            inner = self.helper(l.right, r.left)
            outer = self.helper(l.left, r.right)
            return inner and outer
    
    # using queue in Symmetric Tree
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        queue = list()
        queue.append(root.left)
        queue.append(root.right)
        
        while len(queue) > 0:
            l = queue.pop(0)
            r = queue.pop(0)
            if not l and not r:
                continue
            if not l or not r or l.val != r.val:
                return False
            queue.append(l.left)
            queue.append(r.right)
            queue.append(l.right)
            queue.append(r.left)
            
        return True
    
    # Maximum Depth of Binary Tree
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        res = 0
        queue = list()
        queue.append(root)
        while len(queue) > 0:
            size = len(queue)
            for i in range(size):
                cur = queue.pop(0)
                if cur.left != None:
                    queue.append(cur.left)
                if cur.right != None:
                    queue.append(cur.right)
            res += 1
        
        return res

    # Maximum Depth of N-ary Tree
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        
        return self.helper(root)
    
    def helper(self, root: 'Node') -> int:
        if not root:
            return 0
        
        res = 0
        for cur in root.children:
            res = max(res, self.helper(cur))
        
        return res + 1

    # Minimum Depth of Binary Tree
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        res = 0
        queue = list()
        queue.append(root)
        while len(queue) > 0:
            size = len(queue)
            res += 1
            for i in range(size):
                cur = queue.pop(0)
                if not cur.left and not cur.right:
                    return res
                if cur.left != None:
                    queue.append(cur.left)
                if cur.right != None:
                    queue.append(cur.right)
        return res
        # postorder way
        # if not root:
        #    return 0
        # return self.helper(root)

    def helper(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        leftNode = root.left
        rightNode = root.right
        if not leftNode and rightNode:
            return 1 + self.helper(rightNode)
        if leftNode and not rightNode:
            return 1 + self.helper(leftNode)
        res = 1 + min(self.helper(leftNode), self.helper(rightNode))
        return res
        
    # Count Complete Tree Nodes
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue = list()
        queue.append(root)
        res = 0
        
        while len(queue) > 0:
            size = len(queue)
            for i in range(size):
                cur = queue.pop(0)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res += size
        return res
        # O(logn * logn)
        #
        # if not root:
            # return 0
        # leftNode = root.left
        # rightNode = root.right
        # leftHeight = rightHeight = 0
        # while leftNode:
            # leftNode = leftNode.left
            # leftHeight += 1
        # while rightNode:
            # rightNode = rightNode.right
            # rightHeight += 1
        # if leftHeight == rightHeight:
            # return (2 << leftHeight) - 1
        # return self.countNodes(root.left) + self.countNodes(root.right) + 1
        #

    # Balanced Binary Tree
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        res = self.getHeight(root)
        return res != -1
    def getHeight(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        leftHeight = self.getHeight(root.left)
        if leftHeight == -1:
            return -1
        rightHeight = self.getHeight(root.right)
        if rightHeight == -1:
            return -1
        return -1 if abs(leftHeight - rightHeight) > 1 else 1 + max(leftHeight, rightHeight)
    
    # Binary Tree Paths
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        res = []
        if not root:
            return res
        path = []
        self.traversal(root, path, res)
        return res
    def traversal(self, root:Optional[TreeNode], path: List[int], res: List[str]) -> None:
        path.append(root.val)
        if not root.left and not root.right:
            tmp = ""
            for i in range(len(path) - 1):
                tmp += str(path[i])
                tmp += "->"
            tmp += str(path[-1])
            res.append(tmp)
            return
        if root.left != None:
            self.traversal(root.left, path, res)
            path.pop()
        if root.right != None:
            self.traversal(root.right, path, res)
            path.pop()

    # Sum of Left Leaves
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        leftValue = self.sumOfLeftLeaves(root.left)
        rightValue = self.sumOfLeftLeaves(root.right)
        curValue = 0
        if root.left and (not root.left.left) and (not root.left.right):
            curValue = root.left.val
        sum = curValue + leftValue + rightValue
        return sum
    
    # Path Sum
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        return self.traversal(root, targetSum - root.val)
    def traversal(self, root: Optional[TreeNode], sum: int) -> bool:
        if (not root.left) and (not root.right) and (sum == 0):
            return True
        if (not root.left) and (not root.right):
            return False
        if root.left:
            if self.traversal(root.left, sum - root.left.val):
                return True
        if root.right:
            if self.traversal(root.right, sum - root.right.val):
                return True
        return False
    
    # path sum II
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        res = []
        if not root:
            return res
        path = []
        self.preorder(root, targetSum, path, res)
        return res
    def preorder(self, root: Optional[TreeNode], targetSum: int, path: List[int], res: List[List[int]]) -> None:
        path.append(root.val)
        if not root.left and (not root.right):
            if targetSum - root.val == 0:
                res.append(path[:])
            return
        if root.left:
            self.preorder(root.left, targetSum - root.val, path, res)
            path.pop()
        if root.right:
            self.preorder(root.right, targetSum - root.val, path, res)
            path.pop()
    
    # Construct Binary Tree from Inorder and Postorder Traversal
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not postorder:
            return None
        rootVal = postorder[-1]
        indexOfRoot = inorder.index(rootVal)
        root = TreeNode(rootVal)
        
        leftInorder = inorder[0:indexOfRoot]
        rightInorder = inorder[indexOfRoot + 1:]
        
        leftPostorder = postorder[0: len(leftInorder)]
        rightPostorder = postorder[len(leftInorder): len(postorder) - 1]
        
        root.left = self.buildTree(leftInorder, leftPostorder)
        root.right = self.buildTree(rightInorder, rightPostorder)
        
        return root
    
    # Construct Binary Tree from Preorder and Inorder Traversal
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None
        rootVal = preorder[0]
        rootIndex = inorder.index(rootVal)
        root = TreeNode(rootVal)
        
        leftInorder = inorder[0:rootIndex]
        rightInorder = inorder[rootIndex + 1:]
        
        leftPreorder = preorder[1: len(leftInorder) + 1]
        rightPreorder = preorder[len(leftInorder) + 1:]
        
        root.left = self.buildTree(leftPreorder, leftInorder)
        root.right = self.buildTree(rightPreorder, rightInorder)
        
        return root
    
    # Maximum Binary Tree
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if len(nums) == 1:
            return TreeNode(nums[0])
        
        return self.helper(nums)
    def helper(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        
        root = TreeNode(max(nums))
        maxIndex = nums.index(root.val)
        
        root.left = self.helper(nums[:maxIndex])
        root.right = self.helper(nums[maxIndex + 1:])
        
        return root
    
    # Merge Two Binary Trees
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1:
            return root2
        if not root2:
            return root1
        
        root = TreeNode(root1.val + root2.val)
        root.left = self.mergeTrees(root1.left, root2.left)
        root.right = self.mergeTrees(root1.right, root2.right)
        return root
    
    # Search in a Binary Search Tree
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root or root.val == val:
            return root
        if root.val > val:
            return self.searchBST(root.left, val)
        if root.val < val:
            return self.searchBST(root.right, val)
        
        return None

    # Validate Binary Search Tree
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        cur = None
        def inorder(root: Optional[TreeNode]) -> bool:
            nonlocal cur
            if not root:
                return True
            left = inorder(root.left)
            if not left:
                return False
            if cur and (cur.val >= root.val):
                return False
            cur = root
            right = inorder(root.right)
            return right
        return inorder(root)
    
    # Minimum Absolute Difference in BST
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        res = float('inf')
        pre = None
        def traverse(cur: Optional[TreeNode]) -> None:
            nonlocal res, pre
            if not cur:
                return
            traverse(cur.left)
            if pre:
                res = min(res, cur.val - pre.val)
            pre = cur
            traverse(cur.right)
        traverse(root)
        return res

    # Find Mode in Binary Search Tree
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        res = list()
        count = maxCount = 0
        pre = None
        def helper(cur: Optional[TreeNode]) -> None:
            if not cur:
                return
            nonlocal res, count, maxCount, pre
            helper(cur.left)
            if not pre or pre.val != cur.val:
                count = 1
            else:
                count += 1
            
            if count > maxCount:
                res.clear()
                res.append(cur.val)
                maxCount = count
            elif count == maxCount:
                res.append(cur.val)
            pre = cur
            helper(cur.right)
        helper(root)
        return res
    
    # Lowest Common Ancestor of a Binary Tree
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root == p or root == q or (not root):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        if not left:
            return right
        return left
    
    # Lowest Common Ancestor of a Binary Search Tree
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root
        return None
    
    # Insert into a Binary Search Tree
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            node = TreeNode(val)
            return node
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        return root
    
    # Delete Node in a BST
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return root
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            cur = root.right
            while cur.left:
                cur = cur.left
            cur.left = root.left
            tmp = root
            root = root.right
            del tmp
            return root
        return root
    
    # trim a binary search tree
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root:
            return root
        if root.val > high:
            return self.trimBST(root.left, low, high)
        if root.val < low:
            return self.trimBST(root.right, low, high)
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root
    
    # Convert Sorted Array to BST
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def traverse(nums: List[int], left: int, right: int) -> Optional[TreeNode]:
            if left > right:
                return None
            mid = left + (right - left) // 2
            root = TreeNode(nums[mid])
            root.left = traverse(nums, left, mid - 1)
            root.right = traverse(nums, mid + 1, right)
            return root
        return traverse(nums, 0, len(nums) - 1)

    # Convert BST to Greater Tree
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        pre = 0
        def traverse(cur: Optional[TreeNode]) -> None:
            nonlocal pre
            if not cur:
                return
            traverse(cur.right)
            cur.val += pre
            pre = cur.val
            traverse(cur.left)
        traverse(root)
        return root
    
    # Combinations
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        path = []
        def backtracking(n: int, k: int, startIndex: int):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startIndex, n - k + len(path) + 2):
                path.append(i)
                backtracking(n, k, i + 1)
                path.pop()
        backtracking(n, k, 1)
        return res
    
    # Combination Sum II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        def backtrack(mySum: int, startIndex: int):
            nonlocal candidates, target
            if mySum == target:
                res.append(path[:])
                return
            for i in range(startIndex, len(candidates)):
                if mySum + candidates[i] > target:
                    return
                if i > startIndex and candidates[i] == candidates[i - 1]:
                    continue
                mySum += candidates[i]
                path.append(candidates[i])
                backtrack(mySum, i + 1)
                path.pop()
                mySum -= candidates[i]
        
        candidates.sort()
        backtrack(0, 0)
        return res
    
    # Combination Sum III
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        path = []
        res = []
        def traverse(k: int, n: int, mySum: int, startIndex: int):
            if len(path) == k:
                if mySum == n:
                    res.append(path[:])
                return
            for i in range(startIndex, 9 - k + len(path) + 2):
                mySum += i
                path.append(i)
                traverse(k, n, mySum, i + 1)
                mySum -= i
                path.pop()
        traverse(k, n, 0, 1)
        return res
    
    # Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        if not digits or len(digits) == 0:
            return res
        
        path = ""
        myArr = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
        def backtrack(digits: str, startIndex: int):
            nonlocal path
            if len(digits) == startIndex:
                res.append(path)
                return
            tmp = myArr[int(digits[startIndex])]
            for i in range(0, len(tmp)):
                path += tmp[i]
                backtrack(digits, startIndex + 1)
                path = path[:-1]
        
        backtrack(digits, 0)
        return res
    
    # Palindrome Partitioning
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        def isPalindrome(start: int, end: int) -> bool:
            nonlocal s
            i, j = start, end
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
        def backtrack(startIndex: int):
            nonlocal s
            if startIndex >= len(s):
                res.append(path[:])
                return
            for i in range(startIndex, len(s)):
                if isPalindrome(startIndex, i):
                    path.append(s[startIndex: i + 1])
                else:
                    continue
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res

    # Restore IP Addresses
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        if len(s) > 12:
            return res
        path = []
        def backtrack(startIndex):
            length = len(path)
            if length > 4:
                return
            if length == 4 and startIndex == len(s):
                res.append(".".join(path))
                return
            for i in range(startIndex, len(s)):
                str = s[startIndex: i+1]
                if len(str) > 3 or int(str) > 255:
                    break
                if len(str) > 1 and str[0] == '0':
                    break
                path.append(str)
                backtrack(i+1)
                path.pop()
        backtrack(0)
        return res
    
    # Subsets
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        def backtrack(startIndex: int):
            nonlocal nums
            res.append(path[:])
            if startIndex >= len(nums):
                return
            for i in range(startIndex, len(nums)):
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res
           
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