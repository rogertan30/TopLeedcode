## 说明:

该清单主要针对客户端大厂算法面试。题目是按照[牛客网客户端高频算法](https://www.nowcoder.com/discuss/447791?source_id=profile_create&channel=-2)统计的算法出现频率来排序的。

清单总题目数为72题，其中相同题型大概有3道，类似题目也有几道，实际也就60多题。如果准备面试的时间比较短，建议优先学习出现频率>1次的题目，大概近40题。要达到大厂面试算法要求，要能够在leetcode或牛客网上默写所有算法，并且能够口述算法每一步的作用(大多数的题目我都有备注)。许多算法经典思想，也要做了解，搞不懂的多去leetcode题解区看看。

每道题都有很多种解法，我给出来的题解一般是推荐度比较高或比较经典的解法，题目是以swift实现的，欢迎补充其他语言版本。

## 推荐:

若你想系统学习算法知识，我推荐你先学习[小码哥恋上数据结构与算法](https://github.com/rogertan30/Love-Leetcode)

若你需要系统准备iOS面试，我推荐你看看[iOS面试知识汇总](https://github.com/rogertan30/CodeForJob)

## 数据来源:

[LeetCode字节跳动企业题库高频算法top100](https://leetcode-cn.com/list/xhx0zp1m)

[牛客网客户端高频算法](https://www.nowcoder.com/discuss/447791?source_id=profile_create&channel=-2)

[小码哥恋上数据结构与算法精选面试题](https://juejin.im/post/6844904118100688904)

## 算法题:

[题目下载地址](https://github.com/rogertan30/TopLeedcode/blob/main/LeetCodeTest.playground.zip)

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
        
