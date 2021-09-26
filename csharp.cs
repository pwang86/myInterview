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

    // ransom note
    public bool CanConstruct(string ransomNote, string magazine) {
        int[] record = new int[26];
        foreach (char i in magazine) {
            record[i - 'a']++;
        }
        
        foreach (char i in ransomNote) {
            if (record[i - 'a'] > 0) {
                record[i - 'a']--;
            } else {
                return false;
            }
        }
        
        return true;
    }

    // 3 sum
     public IList<IList<int>> ThreeSum(int[] nums) {
        IList<IList<int>> res = new List<IList<int>>();
        Array.Sort(nums);
        for (int i = 0; i < nums.Length; i++) {
            if (nums[i] > 0) return res;
            
            if (i > 0 && nums[i - 1] == nums[i]) continue;
            
            int left = i + 1;
            int right = nums.Length - 1;
            while (right > left) {
                int tmp = nums[i] + nums[left] + nums[right];
                if (tmp > 0) {
                    right--;
                } else if (tmp < 0) {
                    left++;
                } else {
                    res.Add(new int[] {nums[i], nums[left], nums[right]}); 
                    
                    while (right > left && nums[right] == nums[right - 1]) right--;
                    while (right > left && nums[left] == nums[left + 1]) left++;
                    right--;
                    left++;
                }
            }
        }
        
        return res;
    }

    // 4 sum
     public IList<IList<int>> FourSum(int[] nums, int target) {
        IList<IList<int>> res = new List<IList<int>>();
        Array.Sort(nums);
        
        for (int i = 0; i < nums.Length; i++) {
            if (i > 0 && nums[i - 1] == nums[i]) continue;
            
            for (int j = i + 1; j < nums.Length; j++) {
                if ((j > i + 1) && nums[j - 1] == nums[j]) continue;
                
                int left = j + 1;
                int right = nums.Length - 1;
                while (right > left) {
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum > target) {
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        res.Add(new int[] {nums[i], nums[j], nums[left], nums[right]});
                        while (right > left && nums[right - 1] == nums[right]) right--;
                        while (right > left && nums[left + 1] == nums[left]) left++;
                        left++;
                        right--;
                    }
                }
            }
        }
        return res;
    }

    // reverse string
    public void ReverseString(char[] s) {
        int l = 0, r = s.Length - 1;
        while (l < r) {
            s[l] ^= s[r];
            s[r] ^= s[l];
            s[l] ^= s[r];
            l++;
            r--;
        }
    }

    // reverse string II
    public string ReverseStr(string s, int k) {
        char[] tmp = s.ToCharArray();
        for (int i = 0; i < tmp.Length; i += 2 * k) {
            int start = i;
            int end = Math.Min(start + k - 1, tmp.Length - 1);
            while (start < end) {
                tmp[start] ^= tmp[end];
                tmp[end] ^= tmp[start];
                tmp[start] ^= tmp[end];
                start++;
                end--;
            }
        }
        return new string(tmp);
    }

    // implement strStr()
    public void GetNext(int[] next, string s) {
        int j = -1;
        next[0] = j;
        for (int i = 1; i < s.Length; i++) {
            while (j >= 0 && s[i] != s[j + 1]) {
                j = next[j];
            }
            if (s[i] == s[j + 1]) {
                j++;
            }
            next[i] = j;
        }
    }
    // get next
    public int StrStr(string haystack, string needle) {
        if (needle.Length == 0) {
            return 0;
        }
        int[] next = new int[needle.Length];
        GetNext(next, needle);

        int j = -1;
        for (int i = 0; i< haystack.Length; i++) {
            while (j >= 0 && haystack[i] != needle[j + 1]) {
                j = next[j];
            }
            if (haystack[i] == needle[j + 1]) {
                j++;
            }
            if (j == needle.Length - 1) {
                return i - needle.Length + 1;
            }
        }
        
        return -1;
    }

    // Reverse Words in a string
    public string ReverseWords(string s) {
        StringBuilder res = RemoveSpace(s);
        ReverseString(res, 0, res.Length - 1);
        ReverseEachWord(res);
        return res.ToString();            
    }
    public StringBuilder RemoveSpace(string s) {
        int l = 0, r = s.Length - 1;
        while (l <= r && s[l] == ' ') l++;
        while (l <= r && s[r] == ' ') r--;
        StringBuilder sb = new StringBuilder();
        while (l <= r) {
            if (s[l] != ' ' || sb[sb.Length - 1] != ' ')
                sb.Append(s[l]);
            l++;
        }
        return sb;
    }
    public void ReverseString(StringBuilder sb, int start, int end) {
        while(start < end) {
            sb[start] ^= sb[end];
            sb[end] ^= sb[start];
            sb[start] ^= sb[end];
            start++;
            end--;
        }
    }
    public void ReverseEachWord(StringBuilder sb) {
        int start = 0, end = 1;
        while (start < sb.Length) {
            while (end < sb.Length && sb[end] != ' ')
                end++;
            ReverseString(sb, start, end - 1);
            start = end + 1;
            end = start + 1;
        }
    }

    // Repeated Substring Pattern
    public bool RepeatedSubstringPattern(string s) {
        if (s.Length == 0) return false;
        
        int[] next = new int[s.Length];
        GetNext(next, s);
        
        int len = s.Length;
        if (next[len - 1] != -1 && len % (len - (next[len - 1] + 1)) == 0)
            return true;
        return false;
    }
    public void GetNext(int[] next, string s) {
        int j = -1;
        next[0] = j;
        for (int i = 1; i < s.Length; i++) {
            while (j >= 0 && s[i] != s[j + 1]) {
                j = next[j];
            }
            if (s[i] == s[j + 1]) {
                j++;
            }
            next[i] = j;
        }
    }

    // valid parentheses
    public bool IsValid(string s) {
        Stack<char> stack = new Stack<char>();
        for (int i = 0; i < s.Length; i++) {
            char tmp = s[i];
            if (tmp == '(') {
                stack.Push(')');
            } else if (tmp == '{') {
                stack.Push('}');
            } else if (tmp == '[') {
                stack.Push(']');
            } else if (stack.Count == 0 || stack.Peek() != tmp) {
                return false;
            } else {
                stack.Pop();
            }
        }
        
        return stack.Count == 0;
    }

    // Remove All Adjacent Duplicates In String
     public string RemoveDuplicates(string s) {
        Stack<char> stack = new Stack<char>();
        for (int i = 0; i < s.Length; i++) {
            if (stack.Count == 0 || stack.Peek() != s[i]) {
                stack.Push(s[i]);
            } else {
                stack.Pop();
            }
        }
        string res = "";
        while (stack.Count != 0) {
            res = stack.Pop() + res;
        }
        return res;
    }

    // Evaluate Reverse Polish Notation
    public int EvalRPN(string[] tokens) {
        Stack<int> stack = new Stack<int>();
        for (int i = 0; i < tokens.Length; i++) {
            if (tokens[i] == "+") {
                stack.Push(stack.Pop() + stack.Pop());
            } else if (tokens[i] == "-") {
                stack.Push(-stack.Pop() + stack.Pop());
            } else if (tokens[i] == "*") {
                stack.Push(stack.Pop() * stack.Pop());
            } else if (tokens[i] == "/") {
                int t1 = stack.Pop();
                int t2 = stack.Pop();
                stack.Push(t2 / t1);
            } else {
                stack.Push(int.Parse(tokens[i]));
            }
        }
        return stack.Pop();
    }

    // top k frequent elements
    public int[] TopKFrequent(int[] nums, int k) {
        if (k == nums.Length) {
            return nums;
        }
        /*
        int[] res = new int[k];
        Dictionary<int, int> dic = new Dictionary<int, int>();
        foreach (int num in nums) {
            if (dic.ContainsKey(num)) {
                dic[num]++;
            } else {
                dic.Add(num, 1);   
            }
        }
        
        var sd = new SortedList<int, List<int>>();
        foreach (var tmp in dic) {
            if (sd.ContainsKey(tmp.Value)) {
                sd[tmp.Value].Add(tmp.Key);
            } else {
                sd.Add(tmp.Value, new List<int>(){tmp.Key});
            }
            if (sd.Count > k) {
                sd.RemoveAt(0);
            }
        }
        int i = 0;
        foreach (int key in sd.Keys.Reverse()) {
            if (i < k) {
                 foreach(int num in sd[key]) {
                     res[i] = num;
                     i++;
                 } 
            }
             
        }
        
        return res;
        */
        if (nums.Length == k) return nums;
        Dictionary<int, int> map = new Dictionary<int, int>();
        foreach (var num in nums) {
            if(!map.ContainsKey(num))
                map.Add(num, 0);
            map[num]++;
        }
        var set = new SortedSet<(int frequency, int num)>();
        foreach (var item in map) {
            set.Add((item.Value, item.Key));
            if(set.Count > k)
                set.Remove(set.Min);
        }
        
        var output = new List<int>();
        foreach (var item in set)
            output.Add(item.num);
        return output.ToArray();
    }

    // Binary Tree Preorder Traversal
    public IList<int> PreorderTraversal(TreeNode root) {
        IList<int> res = new List<int>();
        if (root == null) return res;
        // PreOrder(root, res);

        // iterative way
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.Push(root);
        while (stack.Any()) {
            TreeNode tmp = stack.Pop();
            res.Add(tmp.val);
            if (tmp.right != null) stack.Push(tmp.right);
            if (tmp.left != null) stack.Push(tmp.left);
        }
        return res;
    }
    public void PreOrder(TreeNode root, IList<int> res) {
        if (root == null) return;
        res.Add(root.val);
        PreOrder(root.left, res);
        PreOrder(root.right, res);
    }

    // Postorder Traversal
    public IList<int> PostorderTraversal(TreeNode root) {
        IList<int> res = new List<int>();
        if (root == null) return res;
        // Helper(root, res);
        
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.Push(root);
        
        while (stack.Count != 0) {
            TreeNode tmp = stack.Pop();
            res.Add(tmp.val);
            if (tmp.left != null) {
                stack.Push(tmp.left);
            }
            if (tmp.right != null) {
                stack.Push(tmp.right);
            }
        }
        return res.Reverse().ToList();
    }
    public void Helper(TreeNode root, IList<int> res) {
        if (root == null) {
            return;
        } else {
            Helper(root.left, res);
            Helper(root.right, res);
            res.Add(root.val);
        }
    }

    // Inorder Traversal
    public IList<int> InorderTraversal(TreeNode root) {
        IList<int> res = new List<int>();
        if (root == null) return res;
        
        // InOrder(root, res);
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur = root;
        
        while (cur != null || stack.Count != 0) {
            if (cur != null) {
                stack.Push(cur);
                cur = cur.left;
            } else {
                cur = stack.Pop();
                res.Add(cur.val);
                cur = cur.right;
            }
        }
        
        return res;
    }
    public void InOrder(TreeNode root, IList<int> res) {
        if (root == null) return;
        InOrder(root.left, res);
        res.Add(root.val);
        InOrder(root.right, res);
    }
    
    // Binary Tree Level Order Traversal
    public IList<IList<int>> LevelOrder(TreeNode root) {
        IList<IList<int>> res = new List<IList<int>>();
        if (root == null) return res;
        
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        while (queue.Count != 0) {
            int size = queue.Count;
            IList<int> tmp = new List<int>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.Dequeue();
                tmp.Add(cur.val);
                if (cur.left != null) queue.Enqueue(cur.left);
                if (cur.right != null) queue.Enqueue(cur.right);
            }
            res.Add(tmp);
        }
        return res;
    }

    // Binary Tree Level Order Traversal II.
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<IList<int>> LevelOrderBottom(TreeNode root) {
        if (root == null) return res;
        Helper(root, 0);
        return res.Reverse().ToList();
    }
    public void Helper(TreeNode root, int level) {
        if (level >= res.Count) {
            // Console.WriteLine(res.Count);
            res.Add(new List<int>());
        }
        res[level].Add(root.val);
        if (root.left != null) Helper(root.left, level + 1);
        if (root.right != null) Helper(root.right, level + 1);
    }

    // Binary Tree Right Side View
    public IList<int> RightSideView(TreeNode root) {
        IList<int> res = new List<int>();
        if (root == null) return res;
        
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        
        while (queue.Count > 0) {
            int size = queue.Count;
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.Dequeue();
                if (cur.left != null) queue.Enqueue(cur.left);
                if (cur.right != null) queue.Enqueue(cur.right);
                if (i == size - 1) res.Add(cur.val);
            }
            
        }
        return res;
    }

    // Average of Levels in Binary Tree
    public IList<double> AverageOfLevels(TreeNode root) {
        IList<double> res = new List<double>();
        if (root == null) return res;
        
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        
        while (queue.Count > 0) {
            int size = queue.Count;
            double sum = 0;
            for (int i = 0; i < size;i++) {
                TreeNode cur = queue.Dequeue();
                sum += cur.val;
                if (cur.left != null) queue.Enqueue(cur.left);
                if (cur.right != null) queue.Enqueue(cur.right);
            }
            res.Add(sum / size);
        }
        
        return res;
    }

    // Populating Next Right Pointers in Each Node
    public Node Connect(Node root) {
        if (root == null) {
            return root;
        }
    
        Queue<Node> queue = new Queue<Node>();
        queue.Enqueue(root);
        Node pre = null;
        Node cur = null;
        
        while (queue.Count > 0) {
            int size = queue.Count;
            for (int i = 0; i < size; i++) {
                if (i == 0) {
                    pre = queue.Dequeue();
                    cur = pre;
                } else {
                    cur = queue.Dequeue();
                    pre.next = cur;
                    pre = pre.next;
                }
                if (cur.left != null) queue.Enqueue(cur.left);
                if (cur.right != null) queue.Enqueue(cur.right);             
            }
            pre.next = null;
        }
        return root;
    }

    // Invert Binary Tree
    public TreeNode InvertTree(TreeNode root) {
        if (root == null)
            return root;
        
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        while (queue.Count > 0) {
            int size = queue.Count;
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.Dequeue();
                Swap(ref cur.left, ref cur.right);
                if (cur.left != null) queue.Enqueue(cur.left);
                if (cur.right != null) queue.Enqueue(cur.right);
            }
        }
        return root;
    }
    public void Swap(ref TreeNode left, ref TreeNode right) {
        TreeNode tmp = left;
        left = right;
        right = tmp;
    }
    // recursive way
    public TreeNode InvertTree(TreeNode root) {
        if (root == null)
            return root;
        
        Swap(ref root.left, ref root.right);
        InvertTree(root.left);
        InvertTree(root.right);
        return root;
    }

    // Symmetric Tree
    public bool IsSymmetric(TreeNode root) {
        if (root == null)
            return false;
        
        return Helper(root.left, root.right);
    }
    
    public bool Helper(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        else if (left == null && right != null) return false;
        else if (left != null && right == null) return false;
        else if (left.val != right.val) return false;
        else {
            bool outer = Helper(left.left, right.right);
            bool inner = Helper(left.right, right.left);
            return outer && inner;
        }
    }

    // using queue in Symmetric Tree
    public bool IsSymmetric(TreeNode root) {
        if (root == null)
            return true;
        
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root.left);
        queue.Enqueue(root.right);
        
        while (queue.Count > 0) {
            TreeNode l = queue.Dequeue();
            TreeNode r = queue.Dequeue();
            if (l == null && r == null) continue;
            if (l == null || r == null || (l.val != r.val)) 
                return false;
            queue.Enqueue(l.left);
            queue.Enqueue(r.right);
            queue.Enqueue(l.right);
            queue.Enqueue(r.left);
        }
        return true;  
    }

    // Maximum Depth of Binary Tree
    public int MaxDepth(TreeNode root) {
        if (root == null) return 0;
        int res = 0;
        
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        while (queue.Count > 0) {
            int size = queue.Count;
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.Dequeue();
                if (cur.left != null) queue.Enqueue(cur.left);
                if (cur.right != null) queue.Enqueue(cur.right);
            }
            res++;
        }
        return res;
    }

    // Maximum Depth of N-ary Tree
    public int MaxDepth(Node root) {
        if (root == null)
            return 0;
        
        return Helper(root);
    }
    public int Helper(Node root) {
        if (root == null) return 0;
        int res = 0;
        for (int i = 0; i < root.children.Count; i++) {
            res = Math.Max(res, Helper(root.children[i]));
        }
        return 1 + res;
    }

    // Minimum Depth of Binary Tree
    public int MinDepth(TreeNode root) {
        if (root == null) return 0;
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        int res = 0;
        
        while (queue.Count > 0) {
            int size = queue.Count;
            res++;
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.Dequeue();
                if (cur.left == null && cur.right == null) {
                    return res;
                }
                if (cur.left != null)
                    queue.Enqueue(cur.left);
                if (cur.right != null) 
                    queue.Enqueue(cur.right);
            }
        }
        return res;
        // postorder way:
        // if (root == null) return 0;
        // return Helper(root);
    }
    public int Helper(TreeNode root) {
        if (root == null) return 0;
        
        TreeNode leftNode = root.left;
        TreeNode rightNode = root.right;
        if (leftNode != null && rightNode == null) {
            return 1 + Helper(leftNode);
        }
        if (leftNode == null && rightNode != null) {
            return 1 + Helper(rightNode);
        }
        int res = 1 + Math.Min(Helper(leftNode), Helper(rightNode));
        return res;
    }

    // Count Complete Tree Nodes
    public int CountNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        int res = 0;
        
        while (queue.Count > 0) {
            int size = queue.Count;
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.Dequeue();
                if (cur.left != null) queue.Enqueue(cur.left);
                if (cur.right != null) queue.Enqueue(cur.right);
            }
            res += size;
        }
        return res;
        /*
         * O(logn * logn)
         
         if (root == null) {
            return 0;
        }
        TreeNode leftNode = root.left;
        TreeNode rightNode = root.right;
        int leftHeight = 0, rightHeight = 0;
        while (leftNode != null) {
            leftNode = leftNode.left;
            leftHeight++;
        }
        while (rightNode != null) {
            rightNode = rightNode.right;
            rightHeight++;
        }
        if (leftHeight == rightHeight) {
            return (2 << leftHeight) - 1;
        }
        return CountNodes(root.left) + CountNodes(root.right) + 1;
        */
    }

    // Balanced Binary Tree
    public bool IsBalanced(TreeNode root) {
        int res = GetHeight(root);
        return res != -1;
    }
    public int GetHeight(TreeNode root) {
        if (root == null)
            return 0;
        int leftHeight = GetHeight(root.left);
        if (leftHeight == -1)
            return -1;
        int rightHeight = GetHeight(root.right);
        if (rightHeight == -1)
            return -1;
        return Math.Abs(leftHeight - rightHeight) > 1 ? -1 : 1 + Math.Max(leftHeight, rightHeight);
    }

    // Binary Tree Paths
    public IList<string> BinaryTreePaths(TreeNode root) {
        IList<string> res = new List<string>();
        if (root == null) {
            return res;
        }
        List<int> path = new List<int>();
        Traversal(root, path, res);
        return res;
    }
    
    public void Traversal(TreeNode root, List<int> path, IList<string> res) {
        path.Add(root.val);
        if (root.left == null && root.right == null) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < path.Count - 1; i++) {
                sb.Append(path[i]);
                sb.Append("->");
            }
            sb.Append(path[path.Count - 1]);
            res.Add(sb.ToString());
            return;
        }
        if (root.left != null) {
            Traversal(root.left, path, res);
            path.RemoveAt(path.Count - 1);
        }
        if (root.right != null) {
            Traversal(root.right, path, res);
            path.RemoveAt(path.Count - 1);
        }
    }

    // Sum of Left Leaves
    public int SumOfLeftLeaves(TreeNode root) {
        if (root == null) {
           return 0;
        }
        int leftValue = SumOfLeftLeaves(root.left);
        int rightValue = SumOfLeftLeaves(root.right);
        
        int curValue = 0;
        if (root.left != null && root.left.left == null && root.left.right == null) {
            curValue = root.left.val;
        }
        
        int sum = curValue + leftValue + rightValue;
        return sum;
    }

    // Sliding Window Maximum
    public class MyQueue {
        List<int> list = new List<int>();
        public void Add(int val) {
            while (list.Count != 0 && val > list[list.Count - 1]) {
                list.RemoveAt(list.Count - 1);
            }
            list.Add(val);
        }
        public void Pop(int val) {
            if (list.Count != 0 && val == list[0]) {
                list.RemoveAt(0);
            }
        }
        public int Peek() {
            return list[0];
        }
    }
    public int[] MaxSlidingWindow(int[] nums, int k) {
        if (nums.Length == 1) {
            return nums;
        }
        int[] res = new int[nums.Length - k + 1];
        int count = 0;
        MyQueue mq = new MyQueue();
        for (int i = 0; i < k; i++) {
            mq.Add(nums[i]);
        }
        res[count++] = mq.Peek();
        for (int i = k; i < nums.Length; i++) {
            mq.Pop(nums[i - k]);
            mq.Add(nums[i]);
            res[count++] = mq.Peek();
        }
        return res;
    }

    // Path Sum
    public bool HasPathSum(TreeNode root, int targetSum) {
        if (root == null)
            return false;
        return Traversal(root, targetSum - root.val);
    }
    public bool Traversal(TreeNode root, int sum) {
        if (root.left == null && root.right == null && sum == 0) return true;
        if (root.left == null && root.right == null) return false;
        
        if (root.left != null) {
             if (Traversal(root.left, sum - root.left.val)) return true;
        }
        if (root.right != null) {
            if (Traversal(root.right, sum - root.right.val)) return true;
        }
        
        return false;
    }

    // path sum II
    public IList<IList<int>> PathSum(TreeNode root, int targetSum) {
        IList<IList<int>> res = new List<IList<int>>();
        if (root == null) {
            return res;
        }
        IList<int> path = new List<int>();
        Preorder(root, targetSum, path, res);
        return res;
    }
    
    public void Preorder(TreeNode cur, int sum, IList<int> path, IList<IList<int>> res) {
        path.Add(cur.val);
        if (cur.left == null && cur.right == null) {
            if (sum - cur.val == 0) {
                res.Add(new List<int>(path));
            }
            return;
        }
        
        if (cur.left != null) {
            Preorder(cur.left, sum - cur.val, path, res);
            path.RemoveAt(path.Count - 1);
        }
        if (cur.right != null) {
            Preorder(cur.right, sum - cur.val, path, res);
            path.RemoveAt(path.Count - 1);
        }  
    }

    // Construct Binary Tree from Inorder and Postorder Traversal
    public TreeNode BuildTree(int[] inorder, int[] postorder) {
        if (postorder.Length == 0) return null;
        
        int rootVal = postorder[postorder.Length - 1];
        int index = Array.IndexOf(inorder, rootVal);
        TreeNode root = new TreeNode();
        root.val = rootVal;
        
        int[] leftInorder = inorder.Take(index).ToArray();
        int[] rightInorder = inorder.Skip(index + 1).ToArray();
        
        int[] leftPostorder = postorder.Take(leftInorder.Length).ToArray();
        int[] rightPostorder = postorder.Skip(leftInorder.Length).Take(postorder.Length - leftInorder.Length - 1).ToArray();
        
        root.left = BuildTree(leftInorder, leftPostorder);
        root.right = BuildTree(rightInorder, rightPostorder);
        
        return root;
    }

    // Construct Binary Tree from Preorder and Inorder Traversal
    public TreeNode BuildTree(int[] preorder, int[] inorder) {
        if (preorder.Length == 0) return null;
        
        int rootVal = preorder[0];
        int rootIndex = Array.IndexOf(inorder, rootVal);
        
        TreeNode root = new TreeNode(rootVal);
        int[] leftInorder = inorder.Take(rootIndex).ToArray();
        int[] rightInorder = inorder.Skip(rootIndex + 1).ToArray();
        
        int[] leftPreorder = preorder.Skip(1).Take(rootIndex).ToArray();
        int[] rightPreorder = preorder.Skip(1 + rootIndex).ToArray();
        
        root.left = BuildTree(leftPreorder, leftInorder);
        root.right = BuildTree(rightPreorder, rightInorder);
        
        return root;
    }

    // maximum binary tree
     public TreeNode ConstructMaximumBinaryTree(int[] nums) {
        if (nums.Length == 1)
            return new TreeNode(nums[0]);
        return Helper(nums, 0, nums.Length - 1);
    }
    public TreeNode Helper(int[] nums, int leftIndex, int rightIndex) {
        if (leftIndex > rightIndex)
            return null;
        
        int max = 0, maxIndex = 0;
        for (int i = 0; i < nums.Length; i++) {
            if (max < nums[i]) {
                max = nums[i];
                maxIndex = i;
            }
        }
        
        TreeNode root = new TreeNode(max);
        root.left = Helper(nums.Take(maxIndex).ToArray(), 0, maxIndex - 1);
        root.right = Helper(nums.Skip(maxIndex + 1).ToArray(), maxIndex + 1, nums.Length - 1);
        return root;
    }

    // Merge Two Binary Trees
    public TreeNode MergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) return root2;
        if (root2 == null) return root1;
        
        TreeNode root = new TreeNode(root1.val + root2.val);
        root.left = MergeTrees(root1.left, root2.left);
        root.right = MergeTrees(root1.right, root2.right);
        
        return root;
    }

    // Search in a Binary Search Tree
    public TreeNode SearchBST(TreeNode root, int val) {
        if (root == null || root.val == val) return root;
        if (root.val > val) return SearchBST(root.left, val);
        if (root.val < val) return SearchBST(root.right, val);
        
        return null;
    }

    // Validate Binary Search Tree
    public TreeNode cur;
    public bool IsValidBST(TreeNode root) {
        if (root == null) return true;
        
        bool left = IsValidBST(root.left);
        if (!left) return false;
        
        if (cur != null && cur.val >= root.val) {
            return false;
        } 
        cur = root;
        
        bool right = IsValidBST(root.right);
        return right;
    }

    // Minimum Absolute Difference in BST
    public int res = Int32.MaxValue;
    public TreeNode pre = null;
    public int GetMinimumDifference(TreeNode root) {
        Traverse(root);
        return res;
    }
    public void Traverse(TreeNode cur) {
        if (cur == null) return;
        
        Traverse(cur.left);
        if (pre != null) {
            res = Math.Min(res, cur.val - pre.val);
        }
        pre = cur;
        Traverse(cur.right);
    }

    // Find Mode in Binary Search Tree
    public int count = 0;
    public int maxCount = 0;
    public TreeNode pre = null;
    public List<int> res = new List<int>();
    
    public int[] FindMode(TreeNode root) {
        Helper(root);
        return res.ToArray();
    }
    
    public void Helper(TreeNode cur) {
        if (cur == null) return;
        Helper(cur.left);
        if (pre == null || pre.val != cur.val) {
            count = 1;
        } else {
            count++;
        }
        if (count > maxCount) {
            res.Clear();
            maxCount = count;
            res.Add(cur.val);
        } else if (count == maxCount) {
            res.Add(cur.val);
        }
        pre = cur;
        Helper(cur.right);
    }

    // Lowest Common Ancestor of a Binary Tree
    public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q || root == null) return root;
        TreeNode left = LowestCommonAncestor(root.left, p, q);
        TreeNode right = LowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) return root;
        if (left == null) return right;
        return left;
    }

    // Lowest Common Ancestor of a Binary Search Tree
    public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        while (root != null) {
            if (root.val > p.val && root.val > q.val) {
                root = root.left;
            } else if (root.val < p.val && root.val < q.val) {
                root = root.right;
            } else {
                return root;
            }
        }
        return null;
    }

    // Insert into a Binary Search Tree
     public TreeNode InsertIntoBST(TreeNode root, int val) {
        if (root == null) {
            TreeNode node = new TreeNode(val);
            return node;
        }
        if (root.val > val) {
            root.left = InsertIntoBST(root.left, val);
        } else {
            root.right = InsertIntoBST(root.right, val);
        }
        return root;
    }

    // Delete Node in a BST
    public TreeNode DeleteNode(TreeNode root, int key) {
        if (root == null) {
            return root;
        }
        
        if (root.val > key) {
            root.left = DeleteNode(root.left, key);
        } else if (root.val < key) {
            root.right = DeleteNode(root.right, key);
        } else {
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;
            TreeNode cur = root.right;
            while (cur.left != null) {
                cur = cur.left;
            }
            cur.left = root.left; 
            TreeNode tmp = root;
            root = root.right;
            return root;
        }
        return root;
    }

    // trim a binary search tree
    public TreeNode TrimBST(TreeNode root, int low, int high) {
        if (root == null) return root;
        if (root.val < low) {
            return TrimBST(root.right, low, high);
        }
        if (root.val > high) {
            return TrimBST(root.left, low, high);
        }
        root.left = TrimBST(root.left, low, high);
        root.right = TrimBST(root.right, low, high);
        return root;
    }

    // Convert Sorted Array to BST
    public TreeNode SortedArrayToBST(int[] nums) {
        TreeNode root = Traverse(nums, 0, nums.Length - 1);
        return root;
    }
    public TreeNode Traverse(int[] nums, int left, int right) {
        if (left > right) return null;
        int mid = left + (right - left) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = Traverse(nums, left, mid - 1);
        root.right = Traverse(nums, mid + 1, right);
        return root;
    }

    // Convert BST to Greater Tree
    int pre = 0;
    public TreeNode ConvertBST(TreeNode root) {
        if (root == null) return root;
        Traverse(root);
        return root;
    }
    public void Traverse(TreeNode cur) {
        if (cur == null) return;
        Traverse(cur.right);
        cur.val += pre;
        pre = cur.val;
        Traverse(cur.left);
    }

    // Combinations
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<int> path = new List<int>();
    public IList<IList<int>> Combine(int n, int k) {
        Backtracking(n, k, 1);
        return res;
    }
    public void Backtracking(int n, int k, int startIndex) {
        if (path.Count == k) {
            res.Add(new List<int>(path));
            return;
        }
        for (int i = startIndex; i <= n - k + path.Count + 1; i++) {
            path.Add(i);
            Backtracking(n, k, i + 1);
            path.RemoveAt(path.Count - 1);
        }
    }

    // Combination Sum II
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<int> path = new List<int>();
    
    public IList<IList<int>> CombinationSum2(int[] candidates, int target) {
        Array.Sort(candidates);
        Backtrack(candidates, target, 0, 0);
        return res;
    }
    
    public void Backtrack(int[] candidates, int target, int sum, int startIndex) {
        if (target == sum) {
            res.Add(new List<int>(path));
            return;
        }
        for (int i = startIndex; i < candidates.Length && sum + candidates[i] <= target; i++) {
            if (i > startIndex && candidates[i] == candidates[i - 1]) continue;
            sum += candidates[i];
            path.Add(candidates[i]);
            Backtrack(candidates, target, sum , i + 1);
            path.RemoveAt(path.Count - 1);
            sum -= candidates[i];
        }
    }

    // Combination Sum III
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<int> path = new List<int>();
    
    public IList<IList<int>> CombinationSum3(int k, int n) {
        Traverse(k, n, 0, 1);
        return res;
    }
    
    public void Traverse(int k, int n, int sum, int startIndex) {
        if (path.Count == k) {
            if (sum == n) res.Add(new List<int>(path));
            return;
        }
        for (int i = startIndex; i <= 9 - k + path.Count + 1; i++) {
            sum += i;
            path.Add(i);
            Traverse(k, n, sum, i + 1);
            sum -= i;
            path.RemoveAt(path.Count - 1);
        }
    }

    // Letter Combinations of a Phone Number
    public IList<string> res = new List<string>();
    public StringBuilder path = new StringBuilder();
    String[] myArr = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    public IList<string> LetterCombinations(string digits) {
        if (digits == null || digits.Length == 0) return res;
        Backtrack(digits, 0);
        return res;
    }
    public void Backtrack(string digits, int startIndex) {
        if (startIndex == digits.Length) {
            res.Add(path.ToString());
            return;
        }
        String tmp = myArr[digits[startIndex] - '0'];
        for (int i = 0; i < tmp.Length; i++) {
            path.Append(tmp[i]);
            Backtrack(digits, startIndex + 1);
            path.Remove(path.Length - 1, 1);
        }
    }

    // Palindrome Partitioning
    public IList<IList<string>> res = new List<IList<string>>();
    public IList<string> path = new List<string>();
    
    public IList<IList<string>> Partition(string s) {
        Backtrack(s, 0);
        return res;
    }
    public void Backtrack(string s, int startIndex) {
        if (startIndex >= s.Length) {
            res.Add(new List<string>(path));
            return;
        }
        for (int i = startIndex; i < s.Length; i++) {
            if (isPalindrome(s, startIndex, i)) {
                path.Add(s.Substring(startIndex, i - startIndex + 1));
            } else {
                continue;
            }
            Backtrack(s, i + 1);
            path.RemoveAt(path.Count - 1);
        }
    }
    public bool isPalindrome(string s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            if (s[i] != s[j]) return false;
        }
        return true;
    }

    // Restore IP Addresses
        public IList<string> res = new List<string>();
    public bool IsValid(string s, int start, int end) {
        if (start > end) return false;
        if (s[start] == '0' && start != end) {
            return false;
        }
        int num = 0;
        for (int i = start; i <= end; i++) {
            if (s[i] - '0' < 0 || s[i] - '9' > 0) {
                return false;
            } 
            num = num * 10 + (s[i] - '0');
            if (num > 255) return false;
        }
        return true;
    }
    public void Backtrack(string s, int startIndex, int point) {
        if (point == 3) {
            if (IsValid(s, startIndex, s.Length - 1)) {
                res.Add(s);
            }
            return;
        }
        for (int i = startIndex; i < s.Length; i++) {
            if (IsValid(s, startIndex, i)) {
                // s = s.Substring(0, i - startIndex + 1) + "." + s.Substring(i - startIndex + 1);
                string tmp = s.Insert(i + 1, ".");
                point++;
                Backtrack(tmp, i + 2, point);
                point--;
                // s = s.Substring(0, i - startIndex + 1) + s.Substring(i - startIndex + 2);
            } else {
                break;
            }
        }
    }
    public IList<string> RestoreIpAddresses(string s) {
        if (s.Length > 12) return res;
        Backtrack(s, 0, 0);
        return res;
    }

    // subsets
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<int> path = new List<int>();
    
    public IList<IList<int>> Subsets(int[] nums) {
        Backtrack(nums, 0);
        return res;
    }
    public void Backtrack(int[] nums, int startIndex) {
        res.Add(new List<int>(path));
        if (startIndex >= nums.Length) return;
        for (int i = startIndex; i < nums.Length; i++) {
            path.Add(nums[i]);
            Backtrack(nums, i + 1);
            path.RemoveAt(path.Count - 1);
        }
    }

    // Subsets II
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<int> path = new List<int>();
    public IList<IList<int>> SubsetsWithDup(int[] nums) {
        Array.Sort(nums);
        Backtrack(nums, 0);
        return res;
    }
    public void Backtrack(int[] nums, int startIndex) {
        res.Add(new List<int>(path));
        for (int i = startIndex; i < nums.Length; i++) {
            if (i > startIndex && nums[i] == nums[i - 1]) {
                continue;
            }
            path.Add(nums[i]);
            Backtrack(nums, i + 1);
            path.RemoveAt(path.Count - 1);
        }
    }

    // Increasing Subsequences
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<int> path = new List<int>();
    public IList<IList<int>> FindSubsequences(int[] nums) {
        Backtrack(nums, 0);
        return res;
    }
    public void Backtrack(int[] nums, int startIndex) {
        if (path.Count > 1) {
            res.Add(new List<int>(path));
        }
        int[] arr = new int[201];
        for (int i = startIndex; i < nums.Length; i++) {
            if ((path.Count != 0 && nums[i] < path[path.Count - 1]) || arr[nums[i] + 100] == 1) continue;
            arr[nums[i] + 100] = 1;
            path.Add(nums[i]);
            Backtrack(nums, i + 1);
            path.RemoveAt(path.Count - 1);
        }
    }

    // Permutations
    public IList<IList<int>> res = new List<IList<int>>();
    public IList<int> path = new List<int>();
    public bool[] used;
    public IList<IList<int>> Permute(int[] nums) {
        if (nums.Length == 0) return res;
        used = new bool[nums.Length];
        Backtrack(nums);
        return res;
    }
    public void Backtrack(int[] nums) {
        if (path.Count == nums.Length) {
            res.Add(new List<int>(path));
            return;
        }
        for (int i = 0; i < nums.Length; i++) {
            if (used[i]) continue;
            used[i] = true;
            path.Add(nums[i]);
            Backtrack(nums);
            path.RemoveAt(path.Count - 1);
            used[i] = false;
        }
    }

    // Permutations II
    IList<IList<int>> res = new List<IList<int>>();
    IList<int> path = new List<int>();
    public IList<IList<int>> PermuteUnique(int[] nums) {
        Array.Sort(nums);
        bool[] used = new bool[nums.Length];
        Backtrack(nums, used);
        return res;
    }
    public void Backtrack(int[] nums, bool[] used) {
        if (path.Count == nums.Length) {
            res.Add(new List<int>(path));
            return;
        }
        for (int i = 0; i < nums.Length; i++) {
            if (i > 0 && nums[i - 1] == nums[i] && used[i - 1] == false) {
                continue;
            }
            if (used[i] == false) {
                used[i] = true;
                path.Add(nums[i]);
                Backtrack(nums, used);
                path.RemoveAt(path.Count - 1);
                used[i] = false;
            }
        }
    }

    // Reconstruct Itinerary
    public IList<string> res = new List<string>();
    public Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();
    public IList<string> FindItinerary(IList<IList<string>> tickets) {
        foreach (List<string> t in tickets) {
            if (!map.ContainsKey(t[0])) {
                map[t[0]] = new List<string>();
            }
            map[t[0]].Add(t[1]);
        }
        foreach(var m in map.Values) {
            m.Sort();
        }
        foreach (var kvp in map) {
            Console.WriteLine("Key = {0}, Value = {1}", kvp.Key, kvp.Value.Count);
        }
        res.Add("JFK");
        Backtrack(tickets, "JFK");
        return res;
    }
    
    public bool Backtrack(IList<IList<string>> tickets, string start) {
        if (res.Count == tickets.Count + 1) return true;
        if (!map.ContainsKey(start) || map[start].Count == 0) return false;
        
        for (int i = 0; i < map[start].Count; i++) {
            string city = map[start][i];
            map[start].RemoveAt(i);
            res.Add(city);
            if (Backtrack(tickets, city)) {
                return true;
            }
            res.RemoveAt(res.Count - 1);
            map[start].Insert(i, city);
        }
        return false;
    }

    // N queens
    public IList<IList<string>> res = new List<IList<string>>();
    public IList<IList<string>> SolveNQueens(int n) {
        char[][] board = new char[n][];
        for (int i = 0; i < n; i++) {
            board[i] = new char[n];
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = '.';
            }
        }
        Backtrack(board, 0, n);
        return res;
    }
    public void Backtrack(char[][] board, int row, int n) {
        if (row == n) {
            res.Add(ArrToList(board));
            return;
        }
        for (int col = 0; col < n; col++) {
            if (IsValid(row, col, n, board)) {
                board[row][col] = 'Q';
                Backtrack(board, row + 1, n);
                board[row][col] = '.';
            }
        }
    }
    public bool IsValid(int row, int col, int n, char[][] board) {
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
    public IList<string> ArrToList(char[][] board) {
        IList<string> list = new List<string>();
        for (int i = 0; i < board[0].Length; i++) {
            list.Add(string.Join("", board[i]));
        }
        return list;
    }

    // Sudoku Solver
    public void SolveSudoku(char[][] board) {
        Helper(board);
    }
    public bool Helper(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    continue;
                }
                for (int k = 1; k <= 9; k++) {
                    if (IsValid(i, j, k, board)) {
                        board[i][j] = (char)(k +'0');
                        if (Helper(board)) return true;
                        board[i][j] = '.';
                    }
                }
                return false;
            }
        }
        return true;
    }
    
    public bool IsValid(int row, int col, int val, char[][] board) {
        for (int i = 0; i < 9; i++) {
            if (board[row][i] == (char)(val + '0')) return false;
        }
        for (int i = 0; i < 9; i++) {
            if (board[i][col] == (char)(val + '0')) return false;
        }
        int sRow = (row / 3) * 3;
        int sCol = (col / 3) * 3;
        for (int i = sRow; i < sRow + 3; i++) {
            for (int j = sCol; j < sCol + 3; j++) {
                if (board[i][j] == (char)(val + '0')) return false;
            }
        }
        return true;
    }

    // Assign Cookies
    public int FindContentChildren(int[] g, int[] s) {
        Array.Sort(g);
        Array.Sort(s);
        int res = 0;
        int index = s.Length - 1;
        for (int i = g.Length - 1; i >= 0; i--) {
            if (index >= 0 && s[index] >= g[i]) {
                res++;
                index--;
            }
        }
        return res;
    }

    // Wiggle Subsequence
    public int WiggleMaxLength(int[] nums) {
        if (nums.Length <= 1) return nums.Length;
        int res = 1;
        int preDiff = 0;
        int curDiff = 0;
        for (int i = 0; i < nums.Length - 1; i++) {
            curDiff = nums[i + 1] - nums[i];
            if ((preDiff <= 0 && curDiff > 0) || (preDiff >= 0 && curDiff < 0)) {
                res++;
                preDiff = curDiff;
            }
        }
        return res;
    }

    // Maximum Subarray
    public int MaxSubArray(int[] nums) {
        int res = Int32.MinValue;
        int count = 0;
        for (int i = 0; i < nums.Length; i++) {
            count += nums[i];
            if (count > res) res = count;
            if (count <= 0) count = 0;
        }
        return res;
    }

    // Best Time to Buy and Sell Stock II
    public int MaxProfit(int[] prices) {
        int res = 0;
        for (int i = 1; i < prices.Length; i++) {
            res += Math.Max(prices[i] - prices[i - 1], 0);
        }
        return res;
    }

    // Jump Game
    public bool CanJump(int[] nums) {
        int cover = 0;
        if (nums.Length == 1) return true;
        for (int i = 0; i <= cover; i++) {
            cover = Math.Max(i + nums[i], cover);
            if (cover >= nums.Length - 1) return true;
        }
        return false;
    }

    // Jump Game II
    public int Jump(int[] nums) {
        if (nums.Length == 1) return 0;
        int cur = 0;
        int next = 0;
        int res = 0;
        for (int i = 0; i < nums.Length - 1; i++) {
            next = Math.Max(nums[i] + i, next);
            if (i == cur) {
                cur = next;
                res++;
            }
        }
        return res;
    }

    // Maximize Sum Of Array After K Negations
     public int LargestSumAfterKNegations(int[] nums, int k) {
        int[] arr = nums.OrderBy(x => Math.Abs(x)).ToArray();
        for (int i = arr.Length - 1; i >= 0; i--) {
            if (arr[i] < 0 && k > 0) {
                k--;
                arr[i] = -arr[i];
            }
        }
      
        if (k % 2 == 1) arr[0] = -arr[0];
        return arr.Sum();
    }

    // Gas Station
     public int CanCompleteCircuit(int[] gas, int[] cost) {
        int start = 0;
        int curSum = 0;
        int totalSum = 0;
        for (int i = 0; i < gas.Length; i++) {
            curSum += gas[i] - cost[i];
            totalSum += gas[i] - cost[i];
            if (curSum < 0) {
                curSum = 0;
                start = i + 1;
            }
        }
        if (totalSum < 0) return -1;
        return start;
    }

    // Candy
    public int Candy(int[] ratings) {
        int[] candies = new int[ratings.Length];
        Array.Fill(candies, 1);
        for (int i = 1; i < candies.Length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }
        for (int i = candies.Length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = Math.Max(candies[i + 1] + 1, candies[i]);
            }
        }
        return candies.Sum();
    }
} 

// implement queue using stacks
public class MyQueue {
    Stack<int> sIn;
    Stack<int> sOut;
    /** Initialize your data structure here. */
    public MyQueue() {
        sIn = new Stack<int>();
        sOut = new Stack<int>();
    }
    
    /** Push element x to the back of queue. */
    public void Push(int x) {
          sIn.Push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int Pop() {
        if (!sOut.Any()) {
            while (sIn.Any()) {
                sOut.Push(sIn.Pop());
            }
        }
        return (int)sOut.Pop();
    }
    
    
    /** Get the front element. */
    public int Peek() {
        int res = this.Pop();
        sOut.Push(res);
        return res;
    }
    
    /** Returns whether the queue is empty. */
    public bool Empty() {
        return sIn.Any() && sOut.Any();
    }
 }

 // implement stack using queues
 public class MyStack {
    private Queue<int> queue;

    /** Initialize your data structure here. */
    public MyStack() {
        queue = new Queue<int>();
    }
    
    /** Push element x onto stack. */
    public void Push(int x) {
       queue.Enqueue(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int Pop() {
        int count = queue.Count;
        count--;
        while (count > 0) {
            queue.Enqueue(queue.Dequeue());
            count--;
        }
        return queue.Dequeue();
    }
    
    /** Get the top element. */
    public int Top() {       
        int res = this.Pop();
        queue.Enqueue(res);
        return res;
    }
    
    /** Returns whether the stack is empty. */
    public bool Empty() {
        return queue.Count == 0;
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