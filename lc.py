import numpy as np
import collections

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        
        median = None
        
        
        return median 
        
    def myAtoi(self, s):
        #%%
        sign_exist = False
        num_exist = False
        nums = [str(i) for i in range(0,10)] 

        continue_str = nums + [' ', '+', '-']

        integer = ''
        i = 0
        while (i < len(s)) and (s[i] in continue_str):
            print('i',i)
            #print('sign', sign_exist)
            print('num', num_exist)
            #print('si', s[i])
            if s[i] in nums:
                #print('first if i', i)
                integer += s[i]
                num_exist = True
                i += 1
            elif (s[i] == '-' or s[i] == '+'):
                #print('second if i', i)
                if num_exist == False and sign_exist == False:
                    integer += s[i]
                    i += 1
                    sign_exist = True
                else:
                    i = len(s) + 1
                    
            elif s[i] == ' ':
                #print('thrid if i', i)
                if (num_exist+sign_exist == 0):
                    integer += s[i]
                    i += 1
                else:
                    i = len(s) + 1
        print(integer)
        #print(int(integer))
            #%%    
        try:
            result = int(integer)
            if result < -2**31:
                return -2**31
            elif result > 2**31 - 1:
                return 2**31 - 1
            else:
                return result
        except:
            return 0
        
    
    def climbStairs(self, n):
        
        if n == 1:
            return 1
        if n ==2:
            return 2
        else:
            n_climb = [0]*n
            n_climb[0] = 1
            n_climb[1] = 2
            for i in range(2, n):
                #print(i)
                n_climb[i] = n_climb[i-1] + n_climb[i-2]
        
            return n_climb[-1]
        
    def longestCommonSubsequence(self, text1, text2):

        mat_LCS = np.zeros((len(text1)+1, len(text2)+1))
        mat_LCS[0,:] = 0; mat_LCS[:,0] = 0
        
        for i in range(1, len(text1)+1):
            for j in range(1, len(text2)+1):
                if text1[i-1] == text2[j-1]:
                    mat_LCS[i,j] = mat_LCS[i-1, j-1] + 1
                else:
                    mat_LCS[i,j] = max(mat_LCS[i-1,j], mat_LCS[i,j-1])

        return int(np.max(mat_LCS))
    
    def coinChange(self, coins, amount):
        if amount == 0:
            return 0
        
        num_coin = [amount+1] * (amount + 1) 
        num_coin[0] = 0
        for i in range(amount+1):
            for c in coins:
                if i>=c: # amount is greater than that coin value 
                    num_coin[i] = min(num_coin[i], num_coin[i-c]+1)
            

        if num_coin[-1] != (amount+1):
            return num_coin[-1] 
        else:
            return -1
    
    def compress(self, chars):
        print(chars)
        if len(chars) == 1:     
            return len(chars)
        
        compress_str = chars[0]
        len_temp = 1
        for i in range(1, len(chars)):
            #print('char i', chars[i])
            #print('char i-1', chars[i-1])
            if chars[i] == chars[i-1]:
                len_temp += 1
            else:
                if len_temp != 1:
                    compress_str += str(len_temp)
                compress_str += chars[i]
                
                len_temp = 1
        
        # last position same as previous, append last len_temp
        if len_temp != 1:
            compress_str += str(len_temp)
  
        chars[:] = list(compress_str)
        print(chars)
        return len(chars)
    
    def maxArea(self, height):
        '''
        max_area = 0
        
        for i in range(len(height)):
            for j in range(i + 1, len(height)):
                area = min(height[i], height[j]) * abs(i-j)
                if area > max_area:
                    max_area = area
        '''
        i = 0
        j = len(height) - 1
        
        max_area = min(height[i], height[j]) * abs(i-j)
        while i<j:
            if height[i] <= height[j]:
                i += 1
                temp_area = min(height[i], height[j]) * abs(i-j)
                if temp_area > max_area:
                    max_area = temp_area
            if height[i] > height[j]:
                j -= 1
                temp_area = min(height[i], height[j]) * abs(i-j)
                if temp_area > max_area:
                    max_area = temp_area
        
        return max_area 
    
    def letterCombinations(self, digits: str):


        map_dict = {}
        map_dict['2'] = ['a', 'b', 'c']
        map_dict['3'] = ['d', 'e', 'f']
        map_dict['4'] = ['g', 'h', 'i']
        map_dict['5'] = ['j', 'k', 'l']
        map_dict['6'] = ['m', 'n', 'o']
        map_dict['7'] = ['p', 'q', 'r', 's']
        map_dict['8'] = ['t', 'u', 'v']
        map_dict['9'] = ['w', 'x', 'y', 'z']
        
        if len(digits) == 0:
            return []
        if len(digits) == 1 and digits in list(map_dict.keys()):
            return map_dict[digits]
        
        # if digits has lenght at least 2 
        needed_list = []
        for s in digits:
            if s in list(map_dict.keys()):
                needed_list.append(map_dict[s])
                
        def letterCombinations_helper(lst1, lst2):
            
            combined_lst = []
            for a in lst1:
                for b in lst2:
                    combined_lst.append(a+b)
        
            return combined_lst   
        
        lst1 = needed_list[0]
        for j in range(1, len(needed_list)):
            lst1 = letterCombinations_helper(lst1, needed_list[j])

        
        return lst1
    
    def isPalindrome(self, x: int) -> bool:
        
        str_x = str(x)
        if len(str_x) == 1:
            return True 
        
        else:
            
            i = 0
            j = len(str_x) - 1
            while str_x[i] == str_x[j] and i<j:
                i += 1
                j -= 1
                
            if i >= j:
                return True
            else:
                return False
                
    def fractionToDecimal(self, numerator_original: int, denominator_original: int) -> str:
        
        print('numerator_original', numerator_original)
        print('denominator_original', denominator_original)
        numerator = abs(numerator_original)
        print('numerator', numerator)
        denominator = abs(denominator_original)
        print('denomiator', denominator)
        negative_sign = (numerator_original * denominator_original < 0)
        
        if numerator%denominator == 0:
            fraction_result = str(int(numerator/denominator))
            if negative_sign:
                fraction_result = '-' + fraction_result
                
            return fraction_result 
        
        integer_part = str(int(numerator/denominator)) + '.'
   
        decimal_part = ''
        remaining_num_lst = [numerator - int(numerator/denominator) * denominator]
        
        termination = False 
        
        while termination == False:
            print('remaining_num_lst', remaining_num_lst)
            decimal_place = int(remaining_num_lst[-1] * 10 / denominator)
            decimal_part += str(decimal_place)
            print('decimal part', decimal_part)
            
        
            remaining_current = remaining_num_lst[-1] * 10 - decimal_place * denominator
            
            termination = ((remaining_current * 10)%denominator == 0) or (remaining_current in remaining_num_lst)
            print('condition 1', ((remaining_current * 10)%denominator == 0))
            print('condition 2', (remaining_current in remaining_num_lst))
            remaining_num_lst.append(remaining_current)          
            
            if ((remaining_current * 10)%denominator == 0):
                decimal_place = int(remaining_num_lst[-1] * 10 / denominator)
                if decimal_place != 0:
                    decimal_part += str(decimal_place)
            

        
        # find the location of the recurrent part 
        if not (remaining_current * 10)%denominator == 0:
            rep_loc = [k for k,v in enumerate(remaining_num_lst) if v == remaining_num_lst[-1]][0]
        
            decimal_part = decimal_part[0:rep_loc] + '(' + decimal_part[rep_loc:] + ')'
        fraction_result = integer_part + decimal_part
        
        if negative_sign:
            
            fraction_result = '-' + fraction_result
        return fraction_result
    
    def maxProfit(self, prices):   
        
        max_profit = 0
        min_price = max(prices) + 1
        for i in prices:
            if i <= min_price:
                min_price = i
            else:
                temp_profit = i - min_price
                if temp_profit >= max_profit:
                    max_profit = temp_profit
                
        '''
        max_profit = 0
        
        for i in range(len(prices)):
            for j in range(i+1, len(prices)):
                temp_profit = prices[j] - prices[i]
                
                if temp_profit >= max_profit:
                    max_profit = temp_profit
        
        '''
        return max_profit 
    
    def longestPalindrome(self, s: str) -> str:
        # init res
        res = ""
        
        for i in range(len(s)):
            
            # odd -> helper, update
            tmp = self.helper(i, i, s)
            if len(res) < len(tmp):
                res = tmp
                
            # even -> helper, update
            tmp = self.helper(i, i + 1, s)
            if len(res) < len(tmp):
                res = tmp
        
        # return res
        return res
    
    def helper(self, l, r, s):
    
        # if inbound and palindrome, move left left and right right
        while (l >= 0 and r < len(s) and s[l] == s[r]):
            l -= 1
            r += 1
            
        # return 
        return s[l + 1: r]
        
    def palindrome_check(s):
        
        return None
    
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        
        num_fourSumCount = 0 

        dict12 = collections.defaultdict(int) ##
        
        for i in nums1:
            for j in nums2:
                dict12[i+j] += 1
                
        for k in nums3:
            for l in nums4:
                num_fourSumCount += dict12[-(k+l)]
        
        return num_fourSumCount
    
    def fourSum(self, nums, target):
        
        sorted_nums = list(np.sort(nums))
        
        result_fourSum = []
        
        for i in range(len(sorted_nums)):
            #print('i',i)
            for j in range(i+1, len(sorted_nums)):
                #print('j',j)
                sum_12 = sorted_nums[i] + sorted_nums[j]
                k = j + 1
                l = len(sorted_nums) - 1
                while k<l:
                    #print('k',k)
                    #print('l',l)
                    sum_1234 = sum_12 + sorted_nums[k] + sorted_nums[l]
                    result_fourSum_ijkl = [sorted_nums[i], sorted_nums[j], sorted_nums[k], sorted_nums[l]] 
                    
                    if sum_1234 == target:
                        k += 1
                        l -= 1
                        if result_fourSum_ijkl not in result_fourSum:
                            result_fourSum.append(result_fourSum_ijkl)

                    elif sum_1234 > target:
                        l -= 1
                    elif sum_1234 < target:
                        k += 1
                 
        return result_fourSum
    
    def reverse(self, x):
        
        
        str_abs_x = str(abs(x))
        
        i = 0
        j = len(str_abs_x) - 1
        while i<j:
            str_abs_x = str_abs_x[0:i] + str_abs_x[j] + str_abs_x[i+1:j] + str_abs_x[i] + str_abs_x[j+1: len(str_abs_x)]
            i += 1
            j -= 1
        
        rev_x = int(str_abs_x)
        if rev_x >= 2**31-1 :
            rev_x = 0
            
        if x < 0:
            rev_x = -rev_x
        
        return rev_x
    
    def threeSumClosest(self, nums, target):
        
        
        sorted_nums = list(np.sort(nums))
        closest_threeSum = sum(sorted_nums[0:3])
        diff = abs(target - closest_threeSum)
        
        for i in range(len(sorted_nums)):
            print('num i', sorted_nums[i])
            j = i+1
            k = len(sorted_nums) - 1
            while j < k:
                new_threeSum = sorted_nums[i] + sorted_nums[j] + sorted_nums[k]
                print('new sum', new_threeSum)
                new_diff = abs(target - new_threeSum)
                print('new_diff', new_diff)
                if new_threeSum == target: 
                    print('ever equal')
                    return new_threeSum
                elif new_threeSum < target:
                    if new_diff <= diff:
                        closest_threeSum = new_threeSum
                        diff = new_diff
                    j += 1
                    
                elif new_threeSum > target:
                    if new_diff <= diff:
                        closest_threeSum = new_threeSum
                        diff = new_diff
                    k -= 1
                
        
        return closest_threeSum
    
    def threeSum(self, nums):
        
        
        sorted_nums = list(np.sort(nums))
        result_threeSum = []
        
        for i in range(len(sorted_nums)):
            num1 = sorted_nums[i]
            j = i + 1
            k = len(sorted_nums) - 1
            while j<k:
                num2 = sorted_nums[j]
                num3 = sorted_nums[k]
                total = num1 + num2 + num3 
                if total == 0:
                    if [num1, num2, num3] not in result_threeSum:
                        result_threeSum.append([num1, num2, num3])
                    j += 1
                    k -= 1
                elif total < 0:
                    j += 1
                elif total > 0:
                    k -= 1
            
    
        return result_threeSum
    def twoSum(self, nums, target: int):
        
        sorted_nums = list(np.sort(nums))
        
        i = 0
        j = len(sorted_nums) - 1
        
        while i<j:
            #print('i =', i, 'j=', j)
            total = sorted_nums[i] + sorted_nums[j]
            #print('total', total)
            if total == target:
                a = i
                b = j
                i = j + 1 # exit the while
            elif total > target:
                j -= 1
            elif total < target:
                i += 1
            
        result = []
        result.append(nums.index(sorted_nums[a]))
        #print('nums before', nums)
        nums[nums.index(sorted_nums[a])] += max(nums)  # avoid indexing issues e.g., [3,3] target 6
        #print('nums after', nums)
        result.append(nums.index(sorted_nums[b]))
        return result
        
    def groupAnagrams(self, strs):
   
        anagrams_list = []
        
        str_to_set = []
        for s in strs:
            str_to_set.append(list(np.sort(list(s))))
            
        to_be_visited = [i for i in range(len(str_to_set))]
        
        while to_be_visited != []:
            i = to_be_visited[0]
            #print('i=', i)
            #print('str[i]', strs[i])
            to_be_visited.pop(0)

            #print('to_be_visited before i,j', to_be_visited)
            anagrams_list_i =[strs[i]]
            
            item_to_be_removed = []
            for j in to_be_visited:
    
                #print('j=', j)
                #print('str[j]', strs[j])
                #print('str_to_set[j]', str_to_set[j])
                #print('str_to_set[i]', str_to_set[i])
                if str_to_set[j] == str_to_set[i]:
                    anagrams_list_i.append(strs[j])
                    #to_be_visited.pop(0)
                    #print('to_be_visited after j', to_be_visited)
                    item_to_be_removed.append(j)
                    #print('item to be removed', item_to_be_removed)
                #print(i,j)
            to_be_visited = list(set(to_be_visited) - set(item_to_be_removed))
            anagrams_list.append(anagrams_list_i)
      
        return anagrams_list
    
    def subarraySum(self, nums, k) -> int:
        
        num_subsequence = 0
        
        for i, ele in enumerate(nums):
            #print('i', i)
            #print('ele', ele)
            subsum = ele
            if subsum == k:
                num_subsequence += 1
            for j in range(i+1, len(nums)):
                #print('j', j)
                #print('nums[j]', nums[j])
                subsum += nums[j]
                #print('subsum', subsum)
                if subsum == k:
                    num_subsequence += 1
        
        return num_subsequence

    def pivotIndex(self, nums) -> int:
        
        pivot_idx = []
        i=0
        # case: in the middle 
        while not pivot_idx and i <= len(nums)-1:

            left_sum = sum(nums[0:i])
            right_sum = sum(nums[i+1:len(nums)])
            
            if left_sum == right_sum:
                
                pivot_idx.append(i)
                
            i+= 1
        
        left_sum_all_except_last = sum(nums[0:len(nums)-1])
        
        if pivot_idx == [] and left_sum_all_except_last == 0:
            pivot_idx.append(len(nums)-1)
        
        if pivot_idx == []:
            
            pivot_idx.append(-1)

        return pivot_idx[0]
        
    def binary_tree_parenthesis(self, lst, string, left, right):
    
        if left > 0:
            #print('left')
            self.binary_tree_parenthesis(lst, string + '(', left - 1, right)
        if right > 0:
            #print('right')
            self.binary_tree_parenthesis(lst, string + ')', left, right - 1)            
        if left == right == 0:
            #print('both 0')
            lst.append(string)        
            
        return lst 

    def generateParenthesis(self, n: int):

        
        all_parenthesis = self.binary_tree_parenthesis([], '', n, n)
        all_parenthesis_keep = []
        #print('full list', all_parenthesis)        
        for ele in all_parenthesis:
            #print('ele', ele)
            #print('valid or not', self.isValid(ele))
            if self.isValid(ele) == True:
                all_parenthesis_keep.append(ele)
                #print('after removal', all_parenthesis_keep)
                
        return all_parenthesis_keep
    
    def isValid(self, s: str) -> bool:
        if len(s)%2 == 1:
            print('odd number element')
            return False
        
        dict_bracket = {'(':')', '[':']', '{':'}'}
        stack = []
        for ele in s:
            if len(stack) == 0 and ele not in list(dict_bracket.keys()): 
                stack.append(ele)
                break
            elif ele in list(dict_bracket.keys()): # ele is one of the left brackets
                stack.append(ele)
            else: # ele is one of the righ brackets
                if ele == dict_bracket[stack[-1]]:
                    stack.pop()
                else:
                    break
                
        #print(stack)
        return not stack
        
    def convert(self, s, numRows): # not passed with memory error
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
    
        #s = "A"
        #numRows =1

        if numRows == 1:
            #print(s)
            return s
        else:
            #original_s = s
            l = len(s)
            numCols = (int((l+1-numRows)/(2*numRows -1)) + 1)*numRows
     
            loc_mat = np.zeros((numRows, numCols))
            for j in range(numCols):
                rem = j%(numRows-1)
                #print(rem)
                if rem == 0:
                    loc_mat[:,j] = 1
                else:
                    loc_mat[-rem-1, j] = 1
            #print(loc_mat)       
            extended_s = ''
            for c in range(numCols):
                for r in range(numRows):
                    str_loc = c * numRows + r
                    #print(str_loc)
                    if loc_mat[r,c] == 0:
                        extended_s += '$'
                    if loc_mat[r,c] == 1 and s!='':
                        #print(s)
                        extended_s += s[0]
                        s = s[1:]
            #print(extended_s)
    
            converted_s = ''
            for r in range(numRows):
                for c in range(numCols): 
                    if loc_mat[r,c] == 1:
                        str_loc = c * numRows + r
                        #print(str_loc)
                        if str_loc < len(extended_s) and extended_s[str_loc] != '$':
                            converted_s += extended_s[str_loc]
            #print(converted_s)

            return converted_s

class MyHashMap:
    "a hashmap class" 
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dic = {}
        self.myname = 'a hash map'

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        self.dic[key] = value
        return self

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        try:
            return self.dic[key]
        except:
            return -1 

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        try:
            self.dic.pop(key)
        except:
            return -1
        
if __name__ == "__main__":
    
    obj = MyHashMap()
    obj.put(1,1)   
    
    sol = Solution()
    print(sol.myAtoi('-91283472332'))
    '''
    print(sol.climbStairs(4))
    
    text1 = "abcde"; text2 = "ace" 
    print(sol.longestCommonSubsequence(text1, text2))
    
    coins = [5]; amount = 11
    print(sol.coinChange(coins, amount))
    
    chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
    print(sol.compress(chars))
    
    height = [1,8,6,2,5,4,8,3,7]
    print(sol.maxArea(height))
  
    print(sol.letterCombinations('23456'))
    
    print(sol.isPalindrome(-121))
    
    
    numerator = 2
    denominator = 3
    print(sol.fractionToDecimal(numerator, denominator))
    

    prices =  [i for i in range(1,100000)]
    
    print(sol.maxProfit(prices))

    
    s = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    lp = sol.longestPalindrome(s)
    print(lp)

    nums1 = [1,2]; nums2 = [-2,-1]; nums3 = [-1,2]; nums4 = [0,2]
    
    foursumcount = sol.fourSumCount(nums1, nums2, nums3, nums4)
    print(foursumcount)
    
    nums = [2,2,2,2,2]; target = 8
    
    result_fourSum = sol.fourSum(nums, target)
    print(result_fourSum)
    

    s = "ABcdefghijk"
    numRows = 4
    sol_convert = sol.convert(s, numRows)
    #print(sol_convert)


    dict_bracket = {'(':')', '[':']', '{':'}'}
    s = '(())'
    valid_bracket = sol.isValid(s)
    #print(valid_bracket)
    
    gen_paren = sol.generateParenthesis(8)
    #print(len(gen_paren))
    #print(gen_paren)
    
    nums = [-1,-1,-1,1,1,1]
    pivot_idx = sol.pivotIndex(nums)
    #print(pivot_idx)
    
    nums = [1,-1, 0]
    k = 0
    num_subsequence = sol.subarraySum(nums,k)
    print(num_subsequence)
    
    nums = [3,3]
    target = 6
    twosum_sol = sol.twoSum(nums, target)
    print(twosum_sol)
    
    nums = [0,0,0,0]
    threesum_sol = sol.threeSum(nums)
    print(threesum_sol)
    
    nums = [1,1,-1,-1,3]
    target = 3
    closest_threeSum = sol.threeSumClosest(nums, target)
    print(closest_threeSum)
    
    x = -123
    rev_x = sol.reverse(x)
    print(rev_x)
    '''