public class Solution {
    // spiral matrix II
    public int[][] GenerateMatrix(int n) {
        
        int[][] res = new int[n][];
        for (int i = 0; i < n; i++) {
            res[i] = new int[n];
        }
        
        int loop = n / 2;
        int startX = 0, startY = 0;
        int offset = 1;
        int count = 1;
        int mid = n / 2;
        
        while (loop > 0) {
            int i = startX, j = startY;
            
            for (; j < startY + n - offset; j++) {
                res[i][j] = count++;
            }
            
            for(; i < startX + n - offset; i++) {
                res[i][j] = count++;
            }
            
            for(; j > startY; j--) {
                res[i][j] = count++;
            }
            
            for(; i> startX; i--) {
                res[i][j] = count++;
            }
            
            loop--;
            
            startX += 1;
            startY += 1;
            
            offset += 2;
            
        }
        
        if (n % 2 == 1) {
            res[mid][mid] = count;
        }
        
        return res;
    }

    // Remove Linked List Elements
    /**
    * Definition for singly-linked list.
    * public class ListNode {
    *     public int val;
    *     public ListNode next;
    *     public ListNode(int x) { val = x; }
    * }
    */
    public ListNode RemoveElements(ListNode head, int val) {
        if (head == null)
            return head;
        
        ListNode dummy = new ListNode(-1, head);
        ListNode pre = dummy;
        ListNode cur = head;
        
        while (cur != null) {
            if (cur.val == val) {
                pre.next = cur.next;
            } else {
                pre = cur;
            }
            cur = cur.next;
        }
            
        return dummy.next;
    }

    // reverse linked list
    public ListNode ReverseList(ListNode head) {
        ListNode cur = head;
        ListNode pre = null;
        ListNode tmp = null;
        
        while (cur != null) {
            tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    }

    // swap nodes in pairs
    public ListNode SwapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        
        ListNode dummy = new ListNode(0, head);
        ListNode pre = dummy;
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            ListNode tmp = cur.next.next;
            pre.next = cur.next;
            cur.next.next = cur;
            cur.next = tmp;
            pre = cur;
            cur = cur.next;
        }
        
        return dummy.next;
        
    }

    // remove nth node  from end of list
    public ListNode RemoveNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode slow = dummy;
        ListNode fast = dummy;
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
    } 

    // intersetion of two linked lists
    public ListNode GetIntersectionNode(ListNode headA, ListNode headB) {
        ListNode curA = headA;
        ListNode curB = headB;
        
        int lenA = 0, lenB = 0;
        while (curA != null) {
            lenA++;
            curA = curA.next;
        }
        while (curB != null) {
            lenB++;
            curB = curB.next;
        }
        curA = headA;
        curB = headB;
        
        if (lenB > lenA) {
            int tmp = lenB;
            lenB = lenA;
            lenA = tmp;
            ListNode tmpNode = curB;
            curB = curA;
            curA = tmpNode;
        }
        
        int gap = lenA - lenB;
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
    }

    // linked list cycle II
    public ListNode DetectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                ListNode index1 = fast;
                ListNode index2 = head;
                while (index1 != index2) {
                    index1 = index1.next;
                    index2 = index2.next;
                }
                return index2;
            }
        }
        return null;
    }

    // valid anagram
    public bool IsAnagram(string s, string t) {
        if(s.Length != t.Length){
            return false;
        }
        
        int[] record = new int[26];
        foreach (char c in s.ToCharArray()) {
            record[c - 'a'] += 1;
        }
        foreach (char c in t.ToCharArray()) {
            record[c - 'a'] -= 1;
        }
        
        for (int i = 0; i < record.Length; i++) {
            if (record[i] != 0)
                return false;
        }
        
        return true;
    }

    // intersection of two arrays
    public int[] Intersection(int[] nums1, int[] nums2) {
        if (nums1.Length == 0 || nums1 == null || nums2.Length == 0 || nums2 == null)
            return new int[0];
        HashSet<int> set1 = new HashSet<int>(nums1);
        HashSet<int> res = new HashSet<int>();
        
        foreach(int i in nums2) {
            if (set1.Contains(i)) {
                res.Add(i);
            }
        }
        
        int[] resArr = new int[res.Count];
        int index = 0;
        foreach (int i in res) {
            resArr[index++] = i;
        }
        return resArr;
    }

    // happy number
    public bool IsHappy(int n) {
        HashSet<int> set = new HashSet<int>();
        while (n != 1 && !set.Contains(n)) {
            set.Add(n);
            n = GetSum(n);
        }
        return n == 1;
    }
    public int GetSum(int n) {
        int sum = 0;
        while (n > 0) {
            int tmp = n % 10;
            sum += tmp * tmp;
            n /= 10;
        }
        return sum;
    }

    // two sum
    public int[] TwoSum(int[] nums, int target) {
        int[] res = new int[2];

        Dictionary<int, int> dic = new Dictionary<int, int>();
        for (int i = 0; i < nums.Length; i++) {
            int tmp = target - nums[i];
            if (dic.ContainsKey(tmp)) {
                res[0] = i;
                res[1] = dic[tmp];
                return res;
            }
            dic.Add(nums[i], i);
        }
        return res;
   }

   // 4 sum II
   public int FourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Dictionary<int, int> dic = new Dictionary<int, int>();
        int res = 0;
        
        foreach (int i in nums1) {
            foreach (int j in nums2) {
                if (dic.ContainsKey(i + j)) {
                    dic[i + j]++;
                } else {
                    dic.Add(i + j, 1);
                }
            }
        }   
        foreach (int i in nums3) {
            foreach (int j in nums4) {
                if (dic.ContainsKey(0 - i - j)) {
                    res += dic[0 - i - j];
                }
            }
        }
        
        return res;
    }
} 

// Deisgn Linked List
public class ListNode {
    public int val;
    public ListNode next;
    public ListNode(int value) {
        this.val = value;
    }
}
public class MyLinkedList {
    
    private ListNode _head;
    private int _size;
    /** Initialize your data structure here. */
    public MyLinkedList() {
        _head = new ListNode(0);
        _size = 0;
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    public int Get(int index) {
        if (index >= _size || index < 0)
            return -1;
        ListNode cur = _head;
        for (int i = 0; i <= index; i++) {
            cur = cur.next;
        }
        return cur.val;
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    public void AddAtHead(int val) {
        AddAtIndex(0, val);
    }
    
    /** Append a node of value val to the last element of the linked list. */
    public void AddAtTail(int val) {
        AddAtIndex(_size, val);
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    public void AddAtIndex(int index, int val) {
        if (index > _size)
            return;
        if (index < 0)
            index = 0;
        _size++;
        ListNode pre = _head;
        for (int i = 0; i < index; i++) {
            pre = pre.next;
        }
        ListNode cur = new ListNode(val);
        cur.next = pre.next;
        pre.next = cur;
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    public void DeleteAtIndex(int index) {
       if (index < 0 || index >= _size) {
           return;
        }
        _size--;
        ListNode pre = _head;
        for (int i = 0; i < index; i++) {
            pre = pre.next;
        }
        pre.next = pre.next.next;
    }
}