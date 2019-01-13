theory pratical
  imports Main
begin

fun to_set :: "'a list \<Rightarrow> 'a set" where
  "to_set [] = {}" |
  "to_set (x#xs) = {x} \<union> to_set xs"
  
  
lemma
  shows "to_set (xs@ys) = to_set xs \<union> to_set ys"
  apply(induction xs)
   apply(simp)
  apply(simp)
    done
    
lemma to_set_tail [simp]:
  shows "to_set (xs@[x]) = {x} \<union> to_set xs"
  apply(induction xs)
   apply(auto)
done

    
lemma
  shows "to_set (rev xs) = to_set xs"
  apply(induction xs, simp)
  apply(simp)
done
    
lemma
  shows "to_set (filter p xs) = { x. x\<in>to_set xs \<and> p x}"
  apply(induction xs, simp)
  apply(auto)
done
    
fun take :: "nat \<Rightarrow> 'a list \<Rightarrow> 'a list" where
  "take m [] = []" |
  "take m (x#xs) = (case m of 0 \<Rightarrow> [] | Suc m \<Rightarrow> x#take m xs)"
    
    
lemma
  shows "to_set (take m xs) \<subseteq> to_set xs"
  apply(induction m  arbitrary: xs)
   apply(case_tac xs, simp+)
  apply(case_tac xs, auto)
done
  
    
lemma
  shows "length (take m xs) = min (length xs) m"
  apply(induction m arbitrary: xs, simp)
   apply(case_tac xs, simp+)
  apply(case_tac xs, auto)
done
    
lemma
  shows "take m (map f xs) = map f (take m xs)"
  apply(induction m arbitrary: xs, case_tac xs, simp+)
  apply(case_tac xs, auto)
done
    
(*replace the consts command with an implementation*)
fun mem :: "'a \<Rightarrow> 'a list \<Rightarrow> bool" where
  "mem x [] = False" |
  "mem x (y#xs) = (if x=y then True else mem x xs)"
  
  
lemma
  shows "mem e xs \<longleftrightarrow> e \<in> to_set xs"
  apply auto
   apply(induction xs, simp+)
   apply(simp split!: if_split_asm)
  apply(induction xs, auto)
done
    
lemma
  assumes "mem e xs"
  shows "mem e (xs@ys)"
  using assms apply -
  apply(induction xs, simp)
  apply(simp)
done
    
lemma
  assumes "mem e xs"
  shows "mem (f e) (map f xs)"
  using assms apply -
  apply(induction xs, simp)
  apply(simp split!: if_split_asm)
done
    
fun append :: "'a list \<Rightarrow> 'a list \<Rightarrow> 'a list" where
  "append [] xs = xs" |
  "append (x#xs) ys = x#(append xs ys)"
    
fun reverse :: "'a list \<Rightarrow> 'a list" where
  "reverse [] = []" |
  "reverse (x#xs) = (reverse xs) @ [x]"
  
lemma rev_p1 [simp]:
  shows "reverse ( xs @ [x] ) = x# reverse xs"
  apply(induction xs, simp)
  apply auto
done
    
lemma append_p1 [simp]:
  shows "append xs [] = xs"
  apply(induction xs, simp)
  apply(auto)
  done

lemma append_p2 [simp]:
  shows "append xs (ys @ [y]) = (append xs ys) @ [y]"
  apply(induction xs, simp)
  apply auto
done
    
lemma
  shows "reverse (reverse xs) = xs"
  apply(induction xs, simp)
  apply(simp)
done
    
lemma
  shows "reverse (append xs ys) = append (reverse ys) (reverse xs)"
  apply(induction xs, simp)
  apply(simp)
    
(*replace the consts command with an implementation*)
consts concat_map :: "'a list \<Rightarrow> ('a \<Rightarrow> 'b list) \<Rightarrow> 'b list"
  
lemma
  shows "concat_map xs (\<lambda>x. [x]) = xs"
  sorry
    
lemma
  shows "concat_map [x] f = f x"
  sorry
    
lemma
  shows "concat_map (concat_map xs f) g = concat_map xs (\<lambda>y. concat_map (f y) g)"
  sorry
    
datatype 'a rose_tree
  = RTree "'a rose_tree list"
  
consts map :: "('a \<Rightarrow> 'b) \<Rightarrow> 'a rose_tree \<Rightarrow> 'b rose_tree"
  
consts to_list :: "'a rose_tree \<Rightarrow> 'a list"
  
consts mirror :: "'a rose_tree \<Rightarrow> 'a rose_tree"
  
lemma
  shows "to_list (map f rs) = List.map f (to_list rs)"
  sorry
    
lemma
  shows "to_list (mirror rs) = reverse (to_list rs)"
  sorry
    
datatype 'a tree
  = Leaf
  | Node "'a" "'a forest"
and 'a forest
  = Nil
  | Cons "'a tree" "'a forest"
  
consts count_tree :: "'a tree \<Rightarrow> 'a \<Rightarrow> nat"
   count_forest :: "'a forest \<Rightarrow> 'a \<Rightarrow> nat"
   
consts member_tree :: "'a tree \<Rightarrow> 'a \<Rightarrow> bool"
  member_forest :: "'a forest \<Rightarrow> 'a \<Rightarrow> bool"
  
lemma
  shows "(member_tree t x \<longleftrightarrow> (count_tree t x \<noteq> 0)) \<and> (member_forest f x \<longleftrightarrow> (count_forest f x \<noteq> 0))"
  sorry

  
end
