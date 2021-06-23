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