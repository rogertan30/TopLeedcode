### 说明:

该清单主要针对客户端大厂算法面试

题目是按照[牛客网客户端高频算法](https://www.nowcoder.com/discuss/447791?source_id=profile_create&channel=-2)统计的算法出现频率来排序的。

该清单总题目数为72题，其中相同题型大概有3道，类似题目也有几道，实际也就60多题。

如果准备面试的时间比较短，建议优先学习出现频率>1次的题目，大概近40题。

要达到大厂面试算法要求，要能够在leetcode或牛客网上默写所有算法。

若你想系统学习算法知识，我推荐你先学习[小码哥恋上数据结构与算法](https://github.com/rogertan30/Love-Leetcode)

若你需要系统准备iOS面试，我推荐你看看[iOS面试知识汇总](https://github.com/rogertan30/CodeForJob)

### 数据来源:

[LeetCode字节跳动企业题库高频算法top100](https://leetcode-cn.com/list/xhx0zp1m)

[牛客网客户端高频算法](https://www.nowcoder.com/discuss/447791?source_id=profile_create&channel=-2)

[小码哥恋上数据结构与算法精选面试题](https://juejin.im/post/6844904118100688904)



```swift
public class ListNode {
    public var val: Int
    public var next: ListNode?
    public init(_ val: Int) {
        self.val = val
        self.next = nil
    }
}

public class TreeNode {
    public var val: Int
    public var left: TreeNode?
    public var right: TreeNode?
    public init(_ val: Int) {
        self.val = val
        self.left = nil
        self.right = nil
    }
}

/* 1-------------------------------------------------- */

/*
 题号：53.最大子序和
 出现频率：7
 难度：简单
 
 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
 
 输入: [-2,1,-3,4,-1,2,1,-5,4]
 输出: 6
 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
 
 注意：1. result的初始值为第一个元素。
 */
func maxSubArray(_ nums: [Int]) -> Int {
    // 1. 创建变量
    var result = nums[0] // result：连续子数组的最大和
    var sum = 0 // sum：遍历数组元素和
    
    for i in nums {
        // 2. 如果 sum > 0，则说明 sum 对结果有增益效果，则 sum 保留并加上当前遍历数字
        if sum > 0 {
            sum += i
        }
        // 3. 如果 sum <= 0，则说明 sum 对结果无增益效果，需要舍弃，则 sum 直接更新为当前遍历数字
        else {
            sum = i
        }
        // 3. 每次比较 sum 和 ans的大小，将最大值置为result，遍历结束返回结果
        result = result >= sum ? result : sum
    }
    return result
}

/* 2-------------------------------------------------- */

/*
 题号：215.数组中的第k个最大元素
 出现频率：7
 难度：中等
 
 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素
 
 输入: [3,2,1,5,6,4] 和 k = 2
 输出: 5
 */

var curNums = [Int]()

func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
    curNums = nums
    // 1. 对数组进行排序
    sort(begin: 0, end: curNums.count)
    // 2. 获取倒数第k个元素
    return curNums[curNums.count - k]
}

// 1.1 递归调用，实现快速排序
func sort(begin: Int, end: Int) {
    // 1.2 当排序的数量小于一个，则退出。
    if end - begin < 2 {return} //*
    
    // 1.3 进行一次排序，并返回中间值。
    let mid = centerIndex(begin: begin, end: end)
    
    // 1.4 在中间值的左右数组分别再进行排序。
    sort(begin: begin, end: mid)
    sort(begin: mid+1, end: end)
}

func centerIndex(begin: Int, end: Int) -> Int{
    
    var curEnd = end
    var curBegin = begin
    // 1.3.1 备份begin位置元素(随机提取一位作为mid)
    let center = curNums[begin]
    // 1.3.2 最后一位的索引应该减去1
    curEnd -= 1
    
    // 1.3.3 排序
    while curBegin < curEnd {
        // 1.3.4 右侧排序
        while curBegin < curEnd {
            if curNums[curEnd] > center { //右边元素大于center //*
                curEnd -= 1
            } else { //右边元素 <= center
                curNums[curBegin] = curNums[curEnd]
                curBegin += 1
                break
            }
        }
        // 1.3.5 左侧排序
        while curBegin < curEnd {
            if curNums[curBegin] < center {
                curBegin += 1
            } else {
                curNums[curEnd] = curNums[curBegin]
                curEnd -= 1
                break
            }
        }
    }
    
    // 1.3.6 将保存的中间值，存入中间位置。
    curNums[curBegin] = center
    return curBegin // 返回center
}

//方法2
func findKthLargest2(_ nums: [Int], _ k: Int) -> Int {
    
    let len = nums.count
    var left = 0
    var right = len - 1
    let target = len - k
    var arr = nums
    
    while true {
        let index = partition(&arr, left: left, right: right)
        
        if index == target {
            return arr[index]
        } else if index < target {
            left = index + 1
        } else {
            right = index - 1
        }
    }
}

func partition(_ nums: inout [Int], left: Int, right: Int) -> Int {
    if right > left {
        let random = Int.random(in: (left...right))
        nums.swapAt(random, right)
    }
    let pivot = nums[right]
    var j = left
    for i in left..<right {
        if pivot > nums[i] {
            nums.swapAt(i, j)
            j += 1
        }
    }
    nums.swapAt(right, j)
    return j
}

/* 3-------------------------------------------------- */

/*
 题号：21.合并两个有序链表
 出现频率：6
 难度：简单
 
 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的
 */

// 双指针法
func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    var l1 = l1
    var l2 = l2
    var cur: ListNode?
    // 1. 创建虚拟头节点。
    let newList = ListNode.init(-1)
    cur = newList
    // 2. 当l1和l2不为空时，比较大小，拼接。
    while l1 != nil && l2 != nil {
        if l1!.val <= l2!.val {
            cur!.next = l1
            l1 = l1!.next
        } else {
            cur!.next = l2
            l2 = l2!.next
        }
        cur = cur!.next
    }
    // 3. 当l1或l2为空时，拼接另一个链表剩余元素。
    cur!.next = l1 ?? l2
    return newList.next
}

// 递归法
func mergeTwoLists2(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    if (l1 == nil) {
        return l2
    } else if (l2 == nil) {
        return l1
    } else if (l1!.val < l2!.val) {
        l1!.next = mergeTwoLists(l1!.next, l2)
        return l1
    } else {
        l2!.next = mergeTwoLists(l1, l2!.next)
        return l2
    }
}

/* 4-------------------------------------------------- */

/*
 题号：剑指 Offer 09. 用两个栈实现队列
 出现频率：6
 难度：简单
 */

var stack1 = [Int]()
var stack2 = [Int]()

func appendTail(_ value: Int) {
    stack1.append(value)
}

func deleteHead() -> Int {
    if stack2.isEmpty {
        while !stack1.isEmpty {
            stack2.append(stack1.popLast()!) //不能removelast，此函数不会改变stack1的count
        }
    }
    return stack2.isEmpty ? -1 : stack2.popLast()! //记住-1
}

/* 5-------------------------------------------------- */

/*
 题号：206. 反转链表
 出现频率：6
 难度：简单
 
 输入: 1->2->3->4->5->NULL
 输出: 5->4->3->2->1->NULL
 */

func reverseList(_ head: ListNode?) -> ListNode? {
    
    var newHead: ListNode? = nil
    var cur = head
    
    while cur != nil {
        // 1 记录即将断开的节点
        let tmp = cur!.next
        // 2 翻转
        cur!.next = newHead
        // 3 重制
        newHead = cur!
        cur = tmp
    }
    return newHead
}

/* 6-------------------------------------------------- */

/*
 题号：236. 二叉树的最近公共祖先
 出现频率：6
 难度：中等
 */

/*
 解题思路如下：
 1、当root为空时，返回nil
 2、如果root的值和p或者q的值相等时，直接返回root
 3、递归左右子树，用left和right表示递归求出的结果
 4、如果left是空，说明p和q节点都不在左子树，那么结果就在右子树，返回right
 5、如果right是空，说明p和q节点都不在右子树，那么结果就在左子树，返回left
 6、如果left和right都不为空，说明p和q节点分别在左右子树，那么结果就是root
 */

func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
    if root == nil || root === p || root === q { return root } //* === // 赋值
    
    let left = lowestCommonAncestor(root?.left, p, q) // 递归
    let right = lowestCommonAncestor(root?.right, p, q)
    
    if left == nil { return right } //退出条件
    if right == nil { return left }
    return root
}

/* 7-------------------------------------------------- */

/*
 题号：160.相交链表
 出现频率：6
 难度：简单
 */

/*
 题号：剑指 Offer 52. 两个链表的第一个公共节点
 出现频率：1
 难度：简单
 */

func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
    var A = headA
    var B = headB
    // 1. A和B要么相等，要么同时为nil，===表示对象相等。
    while A !== B {
        A = A == nil ? headB : A!.next
        B = B == nil ? headA : B!.next
    }
    return A
}

/* 8-------------------------------------------------- */

/*
 题号：25.k个一组翻转链表
 出现频率：6
 难度：困难
 */

//pre、end、start、next四个游标，移动到group头尾 -> 头尾断开 -> 反转 -> 合并 -> 准备next group

func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
    //1. 虚拟头节点
    let newHead = ListNode.init(-1)
    newHead.next = head
    
    //2. pre end
    var pre: ListNode? = newHead
    var end: ListNode? = newHead
    
    //3. 位移k
    while end != nil {
        for _ in 0..<k {
            if end != nil {
                end = end!.next
            }
        }
        if end == nil { break }
        
        //4. start next
        let start = pre?.next
        let next = end?.next
        
        //5. 断开
        end?.next = nil
        
        //6. 反转
        pre?.next = reverse(head: start!)
        
        //7. 合并
        start?.next = next
        
        //8. 重制
        pre = start
        end = start
    }
    
    return newHead.next
}

// 反转链表
func reverse(head: ListNode) -> ListNode{
    var newHead: ListNode? = nil
    var cur: ListNode? = head
    
    while cur != nil {
        // 5.1 记录即将断开的节点
        let next = cur!.next
        // 5.2 翻转
        cur!.next = newHead
        // 5.3 重制
        newHead = cur
        cur = next
    }
    return newHead! //反转后的头节点
}

/* 9-------------------------------------------------- */

/*
 题号：146.LRU缓存机制
 出现频率：5
 难度：中等
 */

// 1.创建双向链表类
class DLikedNode {
    var pre: DLikedNode?
    var next: DLikedNode?
    var value: Int?
    var key: Int?
    
    // 2. 构造函数
    init(pre: DLikedNode? = nil, next: DLikedNode? = nil, value: Int, key: Int? = nil) {
        self.pre = pre
        self.next = next
        self.value = value
        self.key = key
    }
}

class LRUCache {
    
    // 3. 属性构建
    var capacity = 0
    var count = 0
    let first = DLikedNode.init(value: -1)
    let last = DLikedNode.init(value: -1)
    
    var hash = Dictionary<Int, DLikedNode>()
    
    //4. 实现构造函数
    init(_ capacity: Int) {
        self.capacity = capacity
        first.next = last
        first.pre = nil
        
        last.pre = first
        last.next = nil
    }
    
    // 7
    // 获取
    func get(_ key: Int) -> Int {
        if let node = hash[key] {
            moveToHead(node: node)
            return node.value!
        } else {
            return -1
        }
    }
    
    // 8
    // 写入
    func put(_ key: Int, _ value: Int) {
        if let old = hash[key] {
            old.value = value
            moveToHead(node: old)
        } else {
            let new = DLikedNode.init(value: value, key: key)
            
            if count == capacity {
                removeLastNode()
            }
            moveToHead(node: new)
            hash[key] = new
            count += 1
        }
    }
    
    // 6
    // 删除尾节点
    func removeLastNode() {
        //移除倒数第二个node
        let theNode = last.pre
        theNode?.pre?.next = last
        last.pre = theNode?.pre
        
        count -= 1
        hash[theNode!.key!] = nil
        
        theNode?.pre = nil
        theNode?.next = nil
    }
    
    // 5
    // 移动到头节点
    func moveToHead(node: DLikedNode) {
        
        //取出节点
        let pre = node.pre
        let next = node.next
        pre?.next = next
        next?.pre = pre
        
        //置入头节点
        node.pre = first
        node.next = first.next
        
        first.next = node
        node.next?.pre = node
    }
}

/* 10-------------------------------------------------- */

/*
 题号：958.二叉树的完全性检验
 出现频率：5
 难度：中等
 */

func isCompleteTree(_ root: TreeNode?) -> Bool {
    guard let root = root else { return false }
    var queue = [root]
    var leaf = false
    
    while !queue.isEmpty {
        let node = queue.removeFirst()
        
        // 1）如果某个节点的右子树不为空，则它的左子树必须不为空
        if node.left == nil && node.right != nil {
            return false
        }
        // 2）如果某个节点的右子树为空，则排在它后面的节点必须没有子节点
        if leaf && (node.left != nil || node.right != nil) {
            return false
        }
        // 3）叶子节点
        if node.left == nil || node.right == nil {
            leaf = true
        }
        // 4）队列添加元素
        if node.left != nil {
            queue.append(node.left!)
        }
        if node.right != nil {
            queue.append(node.right!)
        }
    }
    return true
}

/* 11-------------------------------------------------- */

/*
 题号：344.反转字符串
 出现频率：4
 难度：简单
 
 输入：["h","e","l","l","o"]
 输出：["o","l","l","e","h"]
 */

// 双指针，原地交换
func reverseString(_ s: inout [Character]) {
    var l = 0
    var r = s.count - 1
    
    while l < r {
        let cur = s[r]
        s[r] = s[l]
        s[l] = cur
        
        l += 1
        r -= 1
    }
}

/* 12-------------------------------------------------- */

/*
 题号：543.二叉树的直径
 出现频率：4
 难度：简单
 */

var currentDiameter = 0

// 通过计算二叉树的深度，获取二叉树的直径
func diameterOfBinaryTree(_ root: TreeNode?) -> Int {
    let _ = maxDepth(root)
    return currentDiameter
}

func maxDepth(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    
    let leftDepth = maxDepth(root.left)
    let rightDepth = maxDepth(root.right)
    
    // 在计算深度的过程中，更新直径
    currentDiameter = max(currentDiameter, leftDepth + rightDepth)
    
    return max(leftDepth, rightDepth) + 1
}

/* 13-------------------------------------------------- */

/*
 题号：104.二叉树的最大深度
 出现频率：4
 难度：简单
 */

func maxDepth2(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    
    let leftDepth = maxDepth(root.left)
    let rightDepth = maxDepth(root.right)
    
    return max(leftDepth, rightDepth) + 1
}

/* 14-------------------------------------------------- */

/*
 题号：144.二叉树的前序遍历
 出现频率：4
 难度：简单
 */
func preorderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    
    guard let root = root else { return result }
    
    result.append(root.val)
    result += preorderTraversal(root.left)
    result += preorderTraversal(root.right)
    
    return result
}

/* 15-------------------------------------------------- */

/*
 题号：121. 买卖股票的最佳时机
 出现频率：4
 难度：简单
 */

//遍历数组，更新最小值
//根据遍历值与最小值的差，更新最大值
func maxProfit(_ prices: [Int]) -> Int {
    
    if prices.count == 0 { return 0 }
    
    // min & max
    var min = prices[0]
    var max = 0
    
    for i in prices {
        if i < min {
            min = i
        }
        
        if i - min > max {
            max = i - min
        }
    }
    return max
}

/* 16-------------------------------------------------- */

/*
 题号：394. 字符串解码
 出现频率：3
 难度：中等
 */
// https://leetcode-cn.com/problems/decode-string/solution/decode-string-fu-zhu-zhan-fa-di-gui-fa-by-jyd/
func decodeString(_ s: String) -> String {
    var stack = [(Int, String)]()
    var words = ""
    var nums = 0
    for c in s {
        if c == "[" {
            stack.append((nums, words)) // 0 ""
            nums = 0
            words = ""
        } else if c == "]" {
            if let (curMutil, lastRes) = stack.popLast() {
                words = lastRes + String(repeating: words, count: curMutil)
            }
        } else if c.isWholeNumber {
            nums = nums * 10 + c.wholeNumberValue!
        } else {
            words.append(c)
        }
    }
    return words
}

/* 17-------------------------------------------------- */

/*
 题号：102. 二叉树的层序遍历
 出现频率：3
 难度：中等
 */

func levelOrder(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else { return [] }
    
    var result = [[Int]]()
    var queue: [TreeNode] = [root]
    
    while !queue.isEmpty {
        var current = [Int]()
        
        for _ in 0 ..< queue.count {
            let node = queue.removeFirst()
            current.append(node.val)
            
            if node.left != nil {
                queue.append(node.left!)
            }
            
            if node.right != nil {
                queue.append(node.right!)
            }
        }
        result.append(current)
    }
    return result
}

/* 18-------------------------------------------------- */

/*
 题号：199. 二叉树的右视图
 出现频率：3
 难度：中等
 */

func rightSideView(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [Int]()
    var queue: [TreeNode] = [root]
    
    while !queue.isEmpty {
        let count = queue.count
        
        for index in 0 ..< count {
            let node = queue.removeFirst()
            if index == count - 1{
                result.append(node.val)
            }
            
            if node.left != nil {
                queue.append(node.left!)
            }
            
            if node.right != nil {
                queue.append(node.right!)
            }
        }
    }
    return result
}

/* 19-------------------------------------------------- */

/*
 题号：145. 二叉树的后序遍历
 出现频率：3
 难度：困难
 */

func postorderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    guard let root = root else { return result }
    result += postorderTraversal(root.left)
    result += postorderTraversal(root.right)
    result.append(root.val)
    return result
}

/* 20-------------------------------------------------- */

/*
 题号：剑指 Offer 27. 二叉树的镜像
 出现频率：3
 难度：简单
 */

func mirrorTree(_ root: TreeNode?) -> TreeNode? {
    guard let root = root else { return nil }
    let right = mirrorTree(root.right)
    let left = mirrorTree(root.left)
    root.right = left
    root.left = right
    return root
}

func mirrorTree2(_ root: TreeNode?) -> TreeNode? {
    guard root != nil else {
        return nil
    }
    var queue = [TreeNode]()
    queue.append(root!)
    while !queue.isEmpty {
        let node = queue.removeFirst()
        let nodeLeft = node.left
        node.left = node.right
        node.right = nodeLeft
        if node.left != nil {
            queue.append(node.left!)
        }
        if node.right != nil {
            queue.append(node.right!)
        }
    }
    return root
}

/* 21-------------------------------------------------- */

/*
 题号：1. 两数之和
 出现频率：3
 难度：简单
 */

func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
    var dic = Dictionary<Int,Int>()
    
    for (i, v) in nums.enumerated() {
        if dic[target - v] != nil{
            return [i, dic[target - v]!]
        }
        dic[v] = i
    }
    return [-1,-1]
}

/* 22-------------------------------------------------- */

/*
 题号：3. 无重复字符的最长子串
 出现频率：3
 难度：中等
 */

func lengthOfLongestSubstring(_ s: String) -> Int {
    var max = 0
    var arr = [Character]()
    
    for i in s {
        while arr.contains(i) {
            arr.removeFirst()
        }
        arr.append(i)
        
        max = arr.count > max ? arr.count : max
    }
    return max
}

/* 23-------------------------------------------------- */

/*
 题号：142. 环形链表 II
 出现频率：3
 难度：中等
 */

func detectCycle(_ head: ListNode?) -> ListNode? {
    let newHead = ListNode.init(-1)
    newHead.next = head
    
    // 1. 声明快慢指针
    var slow: ListNode? = newHead
    var fast: ListNode? = newHead
    
    // 2. 快慢指针开始移动
    while fast != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        // 3. 找到环，重置慢指针
        if slow === fast {
            slow = newHead
            // 4. 快慢指针一起移动，找到环
            while slow !== fast {
                slow = slow?.next
                fast = fast?.next
            }
            return slow
        }
    }
    return nil
}

/* 24-------------------------------------------------- */

/*
 题号：151. 翻转字符串里的单词
 出现频率：3
 难度：中等
 */

func reverseWords(_ s: String) -> String {
    
    var array = [Character]() // 装填每个单词
    var result = [[Character]]() // 装填所有单词
    
    for item in s {
        // 装填一次单词
        if item == " " && array.count != 0{
            result.append(array)
            array.removeAll()
        }
        // 略过多余空格
        else if item == " " && array.count == 0 {
            continue
        }
        // 添加单词
        else {
            array.append(item)
        }
    }
    // 添加最后一个单词
    if array.count != 0 {result.append(array)}
    
    var str = ""
    let count = result.count
    while result.count != 0 {
        // 非第一个单词，添加一个空格
        if result.count != count {str.append(contentsOf: " ")}
        // 倒叙添加单词
        str.append(contentsOf: result.popLast()!)
    }
    return str
}

/* 25-------------------------------------------------- */

/*
 题号：226. 翻转二叉树
 出现频率：2
 难度：简单
 */

func invertTree(_ root: TreeNode?) -> TreeNode? {
    guard let curRoot = root else { return root }
    
    let tmp = curRoot.left
    curRoot.left = curRoot.right
    curRoot.right = tmp
    
    invertTree(curRoot.left)
    invertTree(curRoot.right)
    
    return root
}

func invertTree2(_ root: TreeNode?) -> TreeNode? {
    if root == nil { return root }
    
    var queue: [TreeNode?] = [root]
    
    while !queue.isEmpty {
        let node = queue.remove(at: 0)
        
        let left = node?.left
        node?.left = node?.right
        node?.right = left
        
        if node?.left != nil {
            queue.append(node?.left)
        }
        
        if node?.right != nil {
            queue.append(node?.right)
        }
    }
    return root
}

func invertTree3(_ root: TreeNode?) -> TreeNode? {
    guard let root = root else { return nil }
    let right = invertTree3(root.right)
    let left = invertTree3(root.left)
    root.right = left
    root.left = right
    return root
}

/* 26-------------------------------------------------- */

/*
 题号：189. 旋转数组
 出现频率：2
 难度：简单
 */

func rotate_00(_ nums: inout [Int], _ k: Int) {
    if nums.count == 0 || nums.count == 1 { return }
    
    var k = k
    
    while k != 0 {
        let last = nums.popLast()!
        nums.insert(last, at: 0)
        k -= 1
    }
}

/* 27-------------------------------------------------- */

/*
 题号：300. 最长上升子序列
 出现频率：2
 难度：中等
 */

func lengthOfLIS(_ nums: [Int]) -> Int {
    // O(n^2) O(n)
    guard nums.count > 0 else { return 0 }
    
    var dp = [Int](repeating: 1, count: nums.count)
    for i in 0..<nums.count {
        for j in 0..<i {
            if nums[i] > nums[j] {
                dp[i] = max(dp[i], dp[j] + 1)
            }
        }
    }
    return dp.max()!
}

/* 28-------------------------------------------------- */

/*
 题号：15. 三数之和
 出现频率：2
 难度：中等
 */

func threeSum(_ nums: [Int]) -> [[Int]] {
    if nums.count < 3 { return [] }
    
    // 0. 排序
    let nums = nums.sorted()
    var result = Set<[Int]>()
    
    
    for i in 0 ..< nums.count {
        
        // 1.最小的数大于0直接跳出循环
        if nums[i] > 0 { break }
        
        // 2.跳过起点相同的
        if i != 0 && nums[i] == nums[i - 1] { continue }
        
        // 3. 初始化左右指针
        var l = i + 1
        var r = nums.count - 1
        
        // 4. 比较
        while l < r {
            let sum = nums[i] + nums[l] + nums[r]
            if sum == 0 {
                result.insert([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
            } else if sum < 0 {
                l += 1
            } else {
                r -= 1
            }
        }
    }
    return Array(result)
}

/* 29-------------------------------------------------- */

/*
 题号：48. 旋转图像
 出现频率：2
 难度：中等
 */

func rotate(_ matrix: inout [[Int]]) {
    let n = matrix.count
    matrix.reverse()
    
    for i in 0 ..< n {
        for j in i ..< n {
            let tmp = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = tmp
        }
    }
}

/* 30-------------------------------------------------- */

/*
 题号：94. 二叉树的中序遍历
 出现频率：2
 难度：中等
 */

func inorderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    guard let root = root else { return result }
    result += inorderTraversal(root.left)
    result.append(root.val)
    result += inorderTraversal(root.right)
    return result
}

/* 31-------------------------------------------------- */

/*
 题号：103. 二叉树的锯齿形层次遍历
 出现频率：2
 难度：中等
 */

/*
 题号：剑指 Offer 32 - III. 从上到下打印二叉树 III
 出现频率：1
 难度：中等
 */

func zigzagLevelOrder(_ root: TreeNode?) -> [[Int]] {
    
    //1. 空判断
    guard let root = root else { return [[Int]]() }
    
    var result = [[Int]]()
    var queue = [root]
    var isZ = false
    
    //2. 创建队列，加入root
    while !queue.isEmpty {
        var cur = [Int]()
        //3. 遍历一次队列
        for _ in 0..<queue.count {
            let node = queue.removeFirst()
            //4. 将队列的值保存
            cur.append(node.val)
            //5. 队列中添加新值
            if node.left != nil {queue.append(node.left!)}
            if node.right != nil {queue.append(node.right!)}
        }
        if isZ { cur.reverse() }
        result.append(cur)
        isZ = !isZ
    }
    return result
}

func zigzagLevelOrder2(_ root: TreeNode?) -> [[Int]] {
    var res: [[Int]] = []
    dfs(root, 0, &res)
    return res
}

private func dfs(_ node: TreeNode?, _ level: Int, _ res: inout [[Int]]) {
    guard let node = node else { return }
    if res.count == level { res.append([]) }
    if level & 1 != 0 {
        res[level].insert(node.val, at: 0)
    } else {
        res[level].append(node.val)
    }
    dfs(node.left, level + 1, &res)
    dfs(node.right, level + 1, &res)
}

/* 32-------------------------------------------------- */

/*
 题号：70. 爬楼梯
 出现频率：2
 难度：简单
 */

func climbStairs(_ n: Int) -> Int {
    var dp: Array = Array.init(repeating: 0, count: n + 1)
    dp[0] = 1
    dp[1] = 1
    
    if n == 1 { return 1 }
    
    for i in 2...n {
        dp[i] = dp[i - 1] + dp[i - 2]
    }
    
    return dp[n]
}

// 空间复杂度 O(1)
func climbStairs2(_ n: Int) -> Int {
    var p = 0
    var q = 0
    var r = 1
    for _ in 0..<n {
        p = q
        q = r
        r = p + q
    }
    return r
}

/* 33-------------------------------------------------- */

/*
 题号：105. 从前序与中序遍历序列构造二叉树
 出现频率：2
 难度：中等
 // https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solution/cong-qian-xu-yu-zhong-xu-bian-li-xu-lie-gou-zao-9/
 */

/*
 题号：剑指 Offer 07. 重建二叉树
 出现频率：1
 难度：中等
 */

var indexMap: [Int: Int] = [:]
// 构造哈希映射，帮助我们快速定位根节点
func buildTree(_ preorder: [Int], _ inorder: [Int]) -> TreeNode? {
    for i in 0..<inorder.count {
        indexMap[inorder[i]] = i
    }
    return myBuildTree(preorder, inorder, 0, preorder.count - 1, 0, inorder.count - 1)
}

func myBuildTree(_ preorder: [Int], _ inorder: [Int], _ preleft: Int, _ preright: Int, _ inleft: Int, _ inright: Int) -> TreeNode? {
    guard preleft <= preright else {
        return nil
    }
    // 前序遍历中的第一个节点就是根节点
    let pre_root = preleft
    // 在中序遍历中定位根节点
    let inroot = indexMap[preorder[pre_root]] ?? 0

    // 先把根节点建立出来
    let root = TreeNode(preorder[pre_root])
    // 得到左子树中的节点数目
    let leftsize = inroot - inleft
    // 递归地构造左子树，并连接到根节点
    // 先序遍历中 从 左边界+1 开始的leftsize个元素就对应了中序遍历中 从左边界开始到根节点定位 -1 的元素
    root.left = myBuildTree(preorder, inorder, preleft + 1, preleft + leftsize, inleft, inroot - 1)
    // 递归地构造右子树，并连接到根节点
    root.right = myBuildTree(preorder, inorder, preleft + leftsize + 1, preright, inroot + 1, inright)
    return root
}

/* 34-------------------------------------------------- */

/*
 题号：190. 颠倒二进制位
 出现频率：2
 难度：简单
 */
func reverseBits(_ n: Int) -> Int {
    var result = 0
    var index = 31
    var n = n
    while index >= 0 {
        result += (n & 1) << index
        n = n >> 1
        index -= 1
    }
    return result
}

/* 35-------------------------------------------------- */

/*
 题号：41. 缺失的第一个正数
 出现频率：2
 难度：困难
 */
func firstMissingPositive(_ nums: [Int]) -> Int {
    guard nums.count > 0 else { return 1 }
    
    var dic = [Int: Int]()
    for i in nums {
        dic[i] = 1
    }
    
    for i in 1...nums.count {
        if dic[i] == nil {
            return i
        }
    }
    
    return nums.count + 1
}

/* 36-------------------------------------------------- */

/*
 题号：54. 螺旋矩阵
 出现频率：2
 难度：中等
 */

/* 37-------------------------------------------------- */

/*
 题号：33. 搜索旋转排序数组
 出现频率：2
 难度：中等
 */

func search(_ nums: [Int], _ target: Int) -> Int {
    
    if nums.count == 0 { return -1 }
    
    var left = 0
    var right = nums.count - 1
    
    // 二分查找
    while left <= right {
        let mid = (left + right) / 2
        
        if nums[mid] == target { return mid }
        
        //判断哪一边有序，在有序的一边判断target的位置。
        if nums[0] <= nums[mid] {
            //通过有序边缩小范围
            if target >= nums[0] && target < nums[mid] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            if target > nums[mid] && target <= nums[right] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    
    return -1
}

/* 38-------------------------------------------------- */

/*
 题号：62. 不同路径
 出现频率：2
 难度：中等
 */

//https://leetcode-cn.com/problems/unique-paths/solution/bu-tong-lu-jing-dong-tai-gui-hua-zu-he-fen-xi-java/
//每一个方格可以由上一个向右或者上一个向下到达
//dp[i][j] = dp[i][j-1] + dp[i-1][j]
func uniquePaths(_ m: Int, _ n: Int) -> Int {
    
    var dp = Array(repeating: Array(repeating: 0, count: m), count: n)
    
    for i in 0..<m { dp[i][0] = 1 }
    for J in 0..<n { dp[0][J] = 1 }
    
    for i in 1..<m {
        for j in 1..<n {
            dp[i][j] = dp[i - 1][j] + dp[i][j-1]
        }
    }
    return dp[m-1][n-1]
}


/* 39-------------------------------------------------- */

/*
 题号：460. LFU缓存
 出现频率：2
 难度：困难
 
 LFU是淘汰一段时间内，使用次数最少的页面。
 */

class LFUCache {
    var capacity: Int // cache capacity
    var count: Int // count of all nodes in the cache
    var min: Int // min reference count among all nodes in the cache
    var nodeMap: [Int: DLNode] // key: node
    var countMap: [Int: DLList] // count: double linked list
    
    init(_ capacity: Int) {
        self.capacity = capacity
        self.count = 0
        self.min = Int.max
        self.nodeMap = [Int: DLNode]()
        self.countMap = [Int: DLList]()
    }
    
    func get(_ key: Int) -> Int {
        if let node = nodeMap[key] {
            updateNode(node)
            return node.value
        } else {
            return -1
        }
    }
    
    func put(_ key: Int, _ value: Int) {
        guard capacity > 0 else {
            return
        }
        if let node = nodeMap[key] {
            node.value = value
            updateNode(node)
        } else {
            // Compare capacity before addition because the new node is guaranteed to evict one of the old nodes.
            // And we don't need to worry about min count because it will be set to 1 as later we are adding a new node.
            if count == capacity {
                if let minList = countMap[min] {
                    let removed = minList.removeLast()
                    nodeMap[removed.key] = nil
                    count -= 1
                }
            }
            let node = DLNode(key, value)
            nodeMap[key] = node
            if let firstList = countMap[1] {
                firstList.add(node)
            } else {
                countMap[1] = DLList(node)
            }
            count += 1
            min = 1
        }
    }
    
    private func updateNode(_ node: DLNode) {
        if let list = countMap[node.count] {
            list.remove(node)
            // If the list after removal is empty, we need to increment `min`.
            if node.count == min && list.isEmpty {
                min += 1
            }
            node.count += 1
            if let newList = countMap[node.count] {
                newList.add(node)
            } else {
                countMap[node.count] = DLList(node)
            }
        }
    }
}

/**
 * Your LFUCache object will be instantiated and called as such:
 * let obj = LFUCache(capacity)
 * let ret_1: Int = obj.get(key)
 * obj.put(key, value)
 */

class DLNode {
    var key: Int
    var value: Int
    var count: Int
    var next: DLNode?
    var prev: DLNode?
    
    init(_ key: Int, _ value: Int) {
        self.key = key
        self.value = value
        self.count = 1
    }
}

class DLList {
    var head: DLNode
    var tail: DLNode
    
    var isEmpty: Bool {
        return head.next === tail && tail.prev === head
    }
    
    init(_ node: DLNode) {
        self.head = DLNode(0, 0)
        self.tail = DLNode(0, 0)
        self.head.next = node
        node.prev = self.head
        node.next = self.tail
        self.tail.prev = node
    }
    
    func add(_ node: DLNode) {
        node.prev = head
        node.next = head.next
        head.next?.prev = node
        head.next = node
    }
    
    func remove(_ node: DLNode) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    func removeLast() -> DLNode {
        // This node must exist.
        let node = tail.prev!
        remove(node)
        return node
    }
}

/* 40-------------------------------------------------- */

/*
 题号：113. 路径总和 II
 出现频率：2
 难度：中等
 
 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
 
 */

var path = [Int]()
var res = [[Int]]()

func pathSum(_ root: TreeNode?, _ sum: Int) -> [[Int]] {
    
    dfs(root, sum)
    return res
}

func dfs(_ root: TreeNode?, _ sum: Int) {
    guard let root = root else { return }
    
    path.append(root.val)
    let tmp = sum - root.val
    if tmp == 0 && root.left == nil && root.right == nil {
        res.append(path)
    }
    
    dfs(root.left, tmp)
    dfs(root.right, tmp)
    
    path.removeLast() // 重点，遍历完后，需要把当前节点remove出去，因为用的是同一个list对象来存所有的路径
}

/* 41-------------------------------------------------- */

/*
 题号：240. 搜索二维矩阵 II
 出现频率：2
 难度：中等
 */

func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
    
    for values in matrix {
        if let lastValue = values.last, lastValue >= target {
            for value in values {
                if value == target {
                    return true
                }
            }
        }
    }
    return false
}

/* 42-------------------------------------------------- */

/*
 题号：101. 对称二叉树
 出现频率：1
 难度：简单
 */

func isSymmetric(_ root: TreeNode?) -> Bool {
    guard let root = root else { return true }
    return dfs(left: root.left, right: root.right)
}

func dfs(left: TreeNode?, right: TreeNode?) -> Bool {
    if left == nil && right == nil {
        return true
    }
    
    if left == nil || right == nil {
        return false
    }
    
    if left!.val != right!.val {
        return false
    }
    
    return dfs(left: left!.left, right: right!.right) && dfs(left: left!.right, right: right!.left)
}

/* 43-------------------------------------------------- */

/*
 题号：136. 只出现一次的数字
 出现频率：1
 难度：简单
 */

func singleNumber(_ nums: [Int]) -> Int {
    var dic = Dictionary<Int, Int>()
    
    for i in nums {
        var count = dic[i]
        count = count == nil ? 1 : count! + 1
        dic[i] = count
    }
    
    for item in dic.keys {
        let value = dic[item]
        if value == 1 { return item }
    }
    return -1
}

/* 44-------------------------------------------------- */

/*
 题号：剑指 Offer 34. 二叉树中和为某一值的路径
 出现频率：1
 难度：中等
 相似题型： 路径总和
 */

var path2 = [Int]()
var res2 = [[Int]]()

func pathSum3(_ root: TreeNode?, _ sum: Int) -> [[Int]] {
    
    dfs(root, sum)
    return res
}

func dfs3(_ root: TreeNode?, _ sum: Int) {
    guard let root = root else { return }
    
    path2.append(root.val)
    let tmp = sum - root.val
    if tmp == 0 && root.left == nil && root.right == nil {
        res2.append(path)
    }
    dfs(root.left, tmp)
    dfs(root.right, tmp)
    path2.removeLast() // 重点，遍历完后，需要把当前节点remove出去，因为用的是同一个list对象来存所有的路径
}

/* 45-------------------------------------------------- */

/*
 题号：328. 奇偶链表
 出现频率：1
 难度：中等
 */

func oddEvenList(_ head: ListNode?) -> ListNode? {
    
    guard let head = head else {
        return nil
    }
    
    var odd: ListNode? = head //奇数
    var even: ListNode? = head.next //偶数
    let evenHead = even //保存头
    
    while even?.next != nil {
        odd!.next = odd!.next!.next
        odd = odd!.next
        even!.next = odd!.next
        even = even!.next
    }
    odd!.next = evenHead
    return head
}

func oddEvenList2(_ head: ListNode?) -> ListNode? {
    var odd = head
    let second = head?.next
    var even = head?.next
    while odd != nil {
        if let nextNode = odd?.next?.next{
            odd?.next = nextNode
            odd = nextNode
            even?.next = nextNode.next
            even = nextNode.next
        }else {
            odd?.next = second
            break
        }
    }
    return head
}

/* 46-------------------------------------------------- */

/*
 题号：162. 寻找峰值
 出现频率：1
 难度：中等
 */
// 首先要注意题目条件，在题目描述中出现了 nums[-1] = nums[n] = -∞，这就代表着 只要数组中存在一个元素比相邻元素大，那么沿着它一定可以找到一个峰值


func findPeakElement(_ nums: [Int]) -> Int {
    var left = 0
    var right = nums.count - 1
    
    while left < right {
        let mid = (right + left) / 2
        if nums[mid] > nums[mid + 1] {
            right = mid
        } else {
            left = mid + 1
        }
    }
    return left
}

/* 47-------------------------------------------------- */

/*
 题号：480. 滑动窗口中位数
 出现频率：1
 难度：困难
 */

/* 48-------------------------------------------------- */

/*
 题号：88. 合并两个有序数组
 出现频率：1
 难度：困难
 */

func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
    
    var current = m + n - 1
    var i1 = m - 1
    var i2 = n - 1
    
    while i2 >= 0 {
        if i1 >= 0 && nums1[i1] >= nums2[i2] {
            nums1[current] = nums1[i1]
            current -= 1
            i1 -= 1
        } else {
            nums1[current] = nums2[i2]
            current -= 1
            i2 -= 1
        }
    }
}

/* 49-------------------------------------------------- */

/*
 题号：322. 零钱兑换
 出现频率：1
 难度：中等
 */

func coinChange(_ coins: [Int], _ amount: Int) -> Int {
    var dp = [Int].init(repeating: Int.max, count: amount+1) // 数组count = amount+1，是因为要多一个dp[0]
    dp[0] = 0
    for i in 0...amount { // 0...amount 为了防止输入coins: [1], amount: 0, 遍历条件变成0...1
        for coin in coins {
            if (coin <= i && dp[i-coin] != Int.max) { // dp[i-coin] != Int.max 为了防止输入coins: [2], amount: 3
                dp[i] = min(dp[i], dp[i-coin]+1)
            }
        }
    }
    return dp[amount] == Int.max ? -1 : dp[amount]
}

/* 50-------------------------------------------------- */

/*
 题号：283. 移动零
 出现频率：1
 难度：简单
 */

//i,j指针同时向前走，当下标i遇到不为0的元素，并且i=j下标（如果相等交不交换没意义，都是同一个数），把下标i的值赋给下标j，把下标i的值赋为0，j下标前进1，一次循环执行完毕。下次循环如果下标i的值等于0，则什么都不做，继续执行下次循环

func moveZeroes(_ nums: inout [Int]) {
    var j = 0
    for (i,_) in nums.enumerated() {
        if nums[i] != 0 {
            if i != j {
                nums[j] = nums[i]
                nums[i] = 0
            }
            j+=1
        }
    }
}

//有点像***的思路, 将第一个0作为锚点, 在O(n)的时间内将不等于0的放左边
func moveZeroes2(_ nums: inout [Int]) {
    var j = 0
    for i in 0..<nums.count {
        //1.遍历数组，如果不为0，则与j替换位置，最终将所有不等于0的数放在数组左侧。
        if nums[i] != 0 {
            nums.swapAt(i, j)
            j += 1
        }
    }
}

/* 51-------------------------------------------------- */

/*
 题号：112. 路径总和
 出现频率：1
 难度：简单
 */

func hasPathSum(_ root: TreeNode?, _ sum: Int) -> Bool {
    //1. 退出条件1
    guard let root = root else { return false }
    
    //2. 退出条件2
    if root.val == sum, root.left == nil, root.right == nil {
        return true
    }
    
    let target = sum - root.val
    //3.递归
    return hasPathSum(root.left, target) || hasPathSum(root.right, target)
}

/* 52-------------------------------------------------- */

/*
 题号：2. 两数相加
 出现频率：1
 难度：中等
 */

func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    var l1 = l1
    var l2 = l2
    
    let head = ListNode.init(-1)
    var cur = head
    var needAppend = 0
    
    while l1 != nil || l2 != nil {
        var l1v = 0
        if l1 != nil {
            l1v = l1!.val
            l1 = l1?.next
        }
        
        var l2v = 0
        if l2 != nil {
            l2v = l2!.val
            l2 = l2?.next
        }
        
        let total = l1v + l2v + needAppend
        
        needAppend = total / 10
        
        cur.next = ListNode.init(total % 10)
        cur = cur.next!
    }
    
    if needAppend == 1{
        cur.next = ListNode.init(1)
    }
    
    return head.next
}

/* 53-------------------------------------------------- */

/*
 题号：257. 二叉树的所有路径
 出现频率：1
 难度：简单
 */

var res3 = [String]()

func binaryTreePaths(_ root: TreeNode?) -> [String] {
    bfs(root, "")
    return res3
}

func bfs(_ root:TreeNode? ,_ s:String) {

    guard let root = root else { return }

    let result = s + "\(root.val)"

    if root.left == nil && root.right == nil {
        res3.append(result)
        return
    }

    bfs(root.left, result + "->")
    bfs(root.right, result + "->")
}

/* 54-------------------------------------------------- */

/*
 题号：46. 全排列
 出现频率：1
 难度：中等
 */

var result = [[Int]]()
var path3 = [Int]()
var used = [Int: Bool]()

func permute(_ nums: [Int]) -> [[Int]] {
    guard nums.count != 0 else {
        return result
    }
    
    dfs(nums: nums, depth: 0)
    return result
}

func dfs(nums: [Int], depth: Int) {
    
    if nums.count == depth {
        result.append(path3)
        return
    }
    
    for i in 0 ..< nums.count {
        if used[nums[i]] ?? false == false {
            path3.append(nums[i])
            used[nums[i]] = true
            
            dfs(nums: nums, depth: depth + 1)
            
            used[nums[i]] = false
            path3.removeLast()
        }
    }
}

/* 55-------------------------------------------------- */

/*
 题号：剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
 出现频率：1
 难度：简单
 */

func exchange(_ nums: [Int]) -> [Int] {
    // 1.创建两个数组, oddNumArr(存储奇数)与 evenNumArr(存储偶数)。
    var oddNumArr = [Int]()
    var evenNumArr = [Int]()
    
    // 2.遍历给定数组中的所有元素, 遇到奇数就存进oddNumArr, 遇到偶数就存进 evenNumArr
    for i in nums {
        if i % 2 == 0 {
            evenNumArr.append(i)
        } else {
            oddNumArr.append(i)
        }
    }
    // 3.最后返回 oddNumArr+evenNumArr
    return oddNumArr + evenNumArr
}

func exchange2(_ nums: [Int]) -> [Int] {
    var nums = nums
    var i = 0
    var j = nums.count - 1
    while i < j {
        while i < j, nums[i] % 2 == 1 {
            i += 1
        }
        while i < j, nums[j] % 2 == 0 {
            j -= 1
        }
        (nums[i], nums[j]) = (nums[j], nums[i])
    }
    return nums
}



/* 56-------------------------------------------------- */

/*
 题号：232. 用栈实现队列
 出现频率：1
 难度：简单
 */

//相同题目 剑指 Offer 09. 用两个栈实现队列


/* 57-------------------------------------------------- */

/*
 题号：234. 回文链表
 出现频率：1
 难度：简单
 */

func isPalindrome(_ head: ListNode?) -> Bool {
    var newHead = head
    var list = [Int]()
    
    // 1.将链表加入到数组中
    while newHead != nil {
        list.append(newHead!.val)
        newHead = newHead?.next
    }
    
    // 2.双指针比较数组两端是否相等
    var start = 0
    var end = list.count - 1
    while start < end {
        if list[start] != list[end] {
            return false
        }
        start += 1
        end -= 1
    }
    return true
}

/* 58-------------------------------------------------- */

/*
 题号：739. 每日温度
 出现频率：1
 难度：中等
 
 可以维护一个存储下标的单调栈，从栈底到栈顶的下标对应的温度列表中的温度依次递减。如果一个下标在单调栈里，则表示尚未找到下一次温度更高的下标。
 
 正向遍历温度列表。对于温度列表中的每个元素 T[i]，如果栈为空，则直接将 i 进栈，如果栈不为空，则比较栈顶元素 prevIndex 对应的温度 T[prevIndex] 和当前温度 T[i]，如果 T[i] > T[prevIndex]，则将 prevIndex 移除，并将 prevIndex 对应的等待天数赋为 i - prevIndex，重复上述操作直到栈为空或者栈顶元素对应的温度小于等于当前温度，然后将 i 进栈。
 
 为什么可以在弹栈的时候更新 ans[prevIndex] 呢？因为在这种情况下，即将进栈的 i 对应的 T[i] 一定是 T[prevIndex] 右边第一个比它大的元素，试想如果 prevIndex 和 i 有比它大的元素，假设下标为 j，那么 prevIndex 一定会在下标 j 的那一轮被弹掉。
 
 由于单调栈满足从栈底到栈顶元素对应的温度递减，因此每次有元素进栈时，会将温度更低的元素全部移除，并更新出栈元素对应的等待天数，这样可以确保等待天数一定是最小的。
 
 */

func dailyTemperatures(_ T: [Int]) -> [Int] {
    var result: [Int] = Array(repeating: 0, count: T.count)
    var stack: [Int] = []
    
    for i in 0 ..< T.count {
        while let index = stack.last, T[i] > T[index] {
            stack.removeLast()
            result[index] = i - index
        }
        stack.append(i)
    }
    return result
}

/* 59-------------------------------------------------- */

/*
 题号：7. 整数反转
 出现频率：1
 难度：简单
 */

func reverse(_ x: Int) -> Int {
    
    var x = x //旧的值
    var result = 0 //新的值
    
    while x != 0 {
        // curX % 10 取模，即最后一位的值
        // curX / 10 除以10，即去掉最后一位
        result = result * 10 + x % 10
        x = x / 10
        
        // 每次转变后检查是否溢出
        if result > Int32.max || result < Int32.min {
            return 0
        }
    }
    return result
}

/* 60-------------------------------------------------- */

/*
 题号：92. 反转链表 II
 出现频率：1
 难度：中等
 */

func reverseBetween(_ head: ListNode?, _ m: Int, _ n: Int) -> ListNode? {
    
    let newHead = ListNode.init(-1)
    newHead.next = head
    
    var pre: ListNode? = newHead
    var cur: ListNode? = head
    
    //移动到反转起始位置
    for _ in 1..<m{
        pre = cur
        cur = cur?.next
    }
     
    let beign = pre//记录反转第一个的前一个
    let end = cur//记录反转的第一个
     
    //反转m到n个元素
    for _ in m...n {
        let next = cur?.next
        cur?.next = pre
        pre = cur
        cur = next
    }
     
    beign?.next = pre//重新标记反转后的头
    end?.next = cur//重新标记反转后的尾
    
    return newHead.next
}

/* 61-------------------------------------------------- */

/*
 题号：662. 二叉树最大宽度
 出现频率：1
 难度：中等
 */
//https://leetcode-cn.com/problems/maximum-width-of-binary-tree/solution/javashuang-bai-yi-chong-you-dian-tou-ji-qu-qiao-de/
//替换成这种算法

func widthOfBinaryTree(_ root: TreeNode?) -> Int {
    
    guard let root = root else { return 0}
    
    var queue: [TreeNode] = [root]
    var list: [Int] = [1]
    var maxLen = 1
    
    while !queue.isEmpty {
        let size = queue.count
        
        for _ in 0..<size {
            let node = queue.removeFirst()
            let index = list.removeFirst()
            
            if let left = node.left {
                queue.append(left)
                list.append(2 &* index)
            }
            if let right = node.right {
                queue.append(right)
                list.append(2 &* index &+ 1)
            }
        }
        
        if list.count >= 2 { //注意临界条件是大于等于2，因为count为1宽度也是1
            maxLen = max(maxLen, list.last! &- list.first! &+ 1)
        }
    }
    return maxLen
}

/* 62-------------------------------------------------- */

/*
 题号：23. 合并K个升序链表
 出现频率：1
 难度：困难
 */
//思路：遍历k个排序链表，记录到字典中，字典的key是链表中的val，字典的value是k个链表中val出现的次数。然后通过字典再生成新的链表。算法的时间复杂度应该是O(nk),n是排序列表中元素的个数。

func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
    
    var result: ListNode?
    var dic = [Int: Int]()
    var cur: ListNode?
    
    // 1.遍历k个排序链表，记录到字典中，字典的key是链表中的val，字典的value是k个链表中val出现的次数
    for node in lists {
        cur = node
        while cur != nil {
            dic.updateValue(dic[cur!.val] ?? 0 + 1, forKey: cur!.val)
            cur = cur!.next
        }
    }
    
    // 2.通过字典再生成新的链表
    for key in dic.keys.sorted() {
        for _ in 0..<dic[key]! {
            if result == nil {
                result = ListNode.init(key)
                cur = result
                continue
            }
            
            cur?.next = ListNode.init(key)
            cur = cur?.next
        }
    }
    
    return result
}


/* 63-------------------------------------------------- */

/*
 题号：141. 环形链表
 出现频率：1
 难度：简单
 */
func hasCycle(_ head: ListNode?) -> Bool {
    
    let newHead = ListNode.init(-1)
    newHead.next = head
    
    // 1. 声明快慢指针
    var slow: ListNode? = newHead
    var fast: ListNode? = newHead
    
    // 2. 快慢指针开始移动
    while fast != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        // 3. 找到环，重置慢指针
        if slow === fast {
            return true
        }
    }
    return false
}

/* 64-------------------------------------------------- */

/*
 题号：140. 单词拆分 II
 出现频率：1
 难度：困难
 */

var map = [String: [String]]()

func wordBreak(_ s: String, _ wordDict: [String]) -> [String] {
    return helper(s, wordDict)
}

func helper(_ s: String, _ wordDict: [String]) -> [String] {
    if s.count == 0 {
        return [""]
    }
    
    if let value = map[s] {
        return value
    }
    
    var res = [String]()
    
    for word in wordDict {
        if s.hasPrefix(word) {
            let subs = helper(String(s[word.endIndex...]), wordDict)
            for sub in subs {
                res.append(word + (sub == "" ? "" : " ") + sub)
            }
        }
    }
    map[s] = res
    
    return res
}

/* 65-------------------------------------------------- */

/*
 题号：86. 分隔链表
 出现频率：1
 难度：中等
 */

// 双指针
func partition(_ head: ListNode?, _ x: Int) -> ListNode? {
        
    var curHead = head
    
    let lHead = ListNode(0)
    var lCur = lHead
    
    let rHead = ListNode(0)
    var rCur = rHead
    
    while curHead != nil {
        if curHead!.val < x {
            lCur.next = curHead
            lCur = curHead!
        } else {
            rCur.next = curHead
            rCur = curHead!
        }
        curHead = curHead?.next
    }
    
    lCur.next = rHead.next
    rCur.next = nil

    return lHead.next
}

/* 66-------------------------------------------------- */

/*
 题号：209. 长度最小的子数组
 出现频率：1
 难度：中等
 
 输入：s = 7, nums = [2,3,1,2,4,3]
 输出：2
 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
 */
func minSubArrayLen(_ s: Int, _ nums: [Int]) -> Int {
    var minLen: Int = .max
    var sum = 0
    var left = 0
    
    for right in 0 ..< nums.count {
        sum += nums[right]
        
        while sum >= s {
            minLen = min(right - left + 1, minLen)
            
            //移除最左边的数，再试图进行一次比较
            sum -= nums[left]
            left += 1
        }
    }
    
    return minLen == .max ? 0 : minLen
}



/* 67-------------------------------------------------- */

/*
 题号：122. 买卖股票的最佳时机 II
 出现频率：1
 难度：简单
 */

//既然不限制交易次数, 那么把能赚钱的交易都累加上就是最大利润了.
func maxProfit2(_ prices: [Int]) -> Int {
    var result = 0
    
    for i in 1..<prices.count {
        let cur = prices[i] - prices[i - 1]
        result += cur > 0 ? cur : 0
    }
    return result
}

/* 68-------------------------------------------------- */

/*
 题号：69. x 的平方根
 出现频率：1
 难度：简单
 */

func mySqrt(_ x: Int) -> Int {
    var l = 0
    var r = x
    var result = 0
    
    while l <= r {
        let mid = l + ( r - l ) / 2
        
        if mid * mid <= x {
            l = mid + 1
            result = mid
        } else {
            r = mid - 1
        }
    }
    
    return result
}

/* 69-------------------------------------------------- */

/*
 题号：128. 最长连续序列
 出现频率：1
 难度：困难
 
 输入: [100, 4, 200, 1, 3, 2]
 输出: 4
 解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
 */

func longestConsecutive(_ nums: [Int]) -> Int {
    // 1.先将数组插入到集合中
    let set = Set(nums)
    
    var maxLength = 0
    for item in set {
        // 2. 遍历集合，如果集合不包含当前元素的上一个，则说明可以从这个元素开始计数(说明没有计过数)
        if !set.contains((item-1)) {
            // 3. 从该数开始计数，如果存在下一个，则+1，否则进入下一次循环
            var next = item + 1
            var curLength = 1 //此次循环最大长度
            while set.contains(next) {
                curLength += 1
                next += 1
            }
            maxLength = max(maxLength, curLength)
        }
    }
    // 4.返回结果即可
    return maxLength
}

/* 70-------------------------------------------------- */

/*
 题号：剑指 Offer 36. 二叉搜索树与双向链表
 出现频率：1
 难度：中等
 https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/solution/mian-shi-ti-36-er-cha-sou-suo-shu-yu-shuang-xian-5/
 */
    var head: TreeNode?, pre: TreeNode?

    func treeToDoublyList(_ root: TreeNode?) -> TreeNode? {
        guard root != nil else {
            return nil
        }
        dfs2(root)
        /** 递归完成后:
         - `pre`已移至最后，可视为`tail`
         */
        head!.left = pre
        pre!.right = head
        return head
    }
    
    /** 递归逻辑:
     - 可视为`pre`不断与`cur/back`双向绑定
     - 转换图形为`Z`，即`中序遍历（LDR）`
     */
    func dfs2(_ root: TreeNode?) {
        /// 结束条件
        guard let root = root else { return }
        
        /// 开始规划左节点，确定前节点`pre`
        dfs2(root.left)
        
        /**:
         1. `pre`节点若不存在则为头结点
         */
        if pre == nil {
            head = root
        }
        
        /** 最下层逻辑:
         1. 从`head`节点开始，前节点`pre`不断上移
         2. 从而双向绑定
         */
        else {
            pre!.right = root
        }
        root.left = pre
        pre = root
        /**:
         - 规划右子树
         - `pre`上移完成后，应对右子树，`pre`需要下移直到右子树递归完成后
         - 后移继续规划左节点
         */
        dfs2(root.right)
    }

/* 71-------------------------------------------------- */

/*
 题号：124. 二叉树中的最大路径和
 出现频率：1
 难度：困难
 */

var pathMax = Int.min

func maxPathSum(_ root: TreeNode?) -> Int {
    dfs(root)
    return pathMax
}

//求一个节点(root)的最大贡献值,具体而言，就是在以该节点为根节点的子树中寻找以该节点为起点的一条路径，使得该路径上的节点值之和最大。
func dfs(_ root: TreeNode?) -> Int{
   
    guard let root = root else { return 0 }
    
    // 递归计算左右子节点的最大贡献值
    // 只有在最大贡献值大于 0 时，才会选取对应子节点
    let leftMax = max(dfs(root.left), 0)
    let rightMax = max(dfs(root.right), 0)
    
    // 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
    // 更新答案
    pathMax = max(pathMax, leftMax + rightMax + root.val)
    
    //返回节点的最大贡献值
    return max(leftMax , rightMax) + root.val
}

/* 72-------------------------------------------------- */

/*
 题号：56. 合并区间
 出现频率：1
 难度：中等
 */

/*
 输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
 输出: [[1,6],[8,10],[15,18]]
 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
 */

func merge(_ intervals: [[Int]]) -> [[Int]] {
    if intervals.count <= 1 { return intervals }
    
    // 1. 排序
    var list = intervals.sorted{ $0[0] < $1[0] }
    var index = 1
    
    while index < list.count {
        // 2. 拿到比较对象
        let pre = list[index - 1]
        let aft = list[index]
        
        // 3.分情况处理
        
        // 3.1 前一个区间包含后一个区间
        if pre.last! > aft.last! {
            list.remove(at: index)
        }
        // 3.2 前一个区间和后一个区间相交
        else if aft.first! <= pre.last! {
            list[index - 1] = [pre.first!, aft.last!]
            list.remove(at: index)
        }
        // 3.3 比较下一位
        else {
            index = index + 1
        }
    }
    return list
}
```
```
