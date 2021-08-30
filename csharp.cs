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