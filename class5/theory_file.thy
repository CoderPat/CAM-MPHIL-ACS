theory theory_file
  imports Main
begin

section\<open>Recursive datatypes and functions\<close>
  
(*some familiar type constructors*)
term "None"
term "Some x"
term "case x of None \<Rightarrow> g | Some y \<Rightarrow> f y"
  
term "Inl x"
term "Inr x"
term "case x of Inl x \<Rightarrow> f x | Inr x \<Rightarrow> g x"

(*the unit constant*)
term "()"

(*pairs*)  
term "(x, y)"
term "fst (x, y)"
value "snd (x, y)"
  
subsection\<open>Lists\<close>

(*lists are already implemented as part of the Isabelle library...*)
term "[]"
term "[1,2,3]"
  
datatype 'a seq
  = Empty ("\<bullet>") (*some syntactic sugar*)
  | Cons 'a "'a seq" (infixr "\<triangleright>" 65) (*declaring \<triangleright> to be right-infix sugar for Cons*)
    
(*definitions are restricted to non-recursive functions*)
definition singleton :: "'a \<Rightarrow> 'a seq" ("\<lbrace>_\<rbrace>" [65]65) where
  "singleton x \<equiv> x \<triangleright> \<bullet>" (* note definitions do not support top-level pattern matching*)
  
(*each definition has an accompanying defining theorem*)
thm singleton_def (*name is name of defined constant with "_def" appended*)
  
(*some hopefully familiar recursive functions*)
fun map :: "('a \<Rightarrow> 'b) \<Rightarrow> 'a seq \<Rightarrow> 'b seq" where
  "map f \<bullet> = \<bullet>" |
  "map f (x \<triangleright> xs) = f x \<triangleright> map f xs"
  
fun app :: "'a seq \<Rightarrow> 'a seq \<Rightarrow> 'a seq" where
  "app \<bullet> ys = ys" |
  "app (x \<triangleright> xs) ys = x \<triangleright> (app xs ys)"
  
fun fold_right :: "('a \<Rightarrow> 'b \<Rightarrow> 'b) \<Rightarrow> 'a seq \<Rightarrow> 'b \<Rightarrow> 'b" where
  "fold_right f \<bullet> e = e" |
  "fold_right f (x \<triangleright> xs) e = f x (fold_right f xs e)"
  
(*note in all cases, Isabelle spots that the functions are terminating automatically*)
  
(*another definition!*)
definition flatten :: "'a seq seq \<Rightarrow> 'a seq" where
  "flatten xss \<equiv> fold_right app xss \<bullet>"
  
thm flatten_def
  
fun len :: "'a seq \<Rightarrow> nat" where
  "len \<bullet> = 0" |
  "len (x \<triangleright> xs) = 1 + len xs"
  
(*recursive functions give rise to inductive proofs*)
  
(*properties about these simple recursive functions*)
lemma
  shows "len (app xs ys) = len xs + len ys"
  apply(induction xs)
   apply simp
  apply simp
  done
    
lemma len_app [simp]:
  shows "len (app xs ys) = len xs + len ys"
  by(induction xs, auto)
    
lemma len_map:
  shows "len (map f xs) = len xs"
  by(induction xs, auto)
    
lemma map_id_ext:
  assumes "\<And>x. f x = x"
  shows "map f xs = xs"
  apply(induction xs)
   apply simp
  apply simp
  apply(rule assms)
  done
    
(*corollary, lemma, and theorem keywords all have the same meaning (to Isabelle, at least)*)
corollary map_id [simp]:
  shows "map id xs = xs" (*"id" is the identity function*)
  apply(rule map_id_ext)
  apply simp
  done
    
lemma map_comp [simp]:
  shows "map f (map g xs) = map (f o g) xs" (*"f o g" is function composition*)
  apply(induction xs)
   apply simp
  apply simp
  done
    
(*note: Isabelle does not unfold definitions automatically when simplifying, defining theorems must
  be manually unfolded or added to the simpset when proving theorems about definitions*)
lemma flatten_Nil [simp]:
  shows "flatten \<bullet> = \<bullet>"
  by(simp add: flatten_def)
    
lemma flatten_Cons [simp]:
  shows "flatten (x \<triangleright> xss) = app x (flatten xss)"
  by(simp add: flatten_def)
    
lemma app_assoc [simp]:
  shows "app xs (app ys zs) = app (app xs ys) zs"
  by(induction xs, auto)
    
(*remove the three lemmas above to appreciate what the simplifier is doing below*)
lemma flatten_app [simp]:
  shows "flatten (app xss yss) = app (flatten xss) (flatten yss)"
  apply(induction xss)
   apply simp
  apply simp
  done
    
subsection\<open>Binary trees\<close>
  
(*slightly more complex examples*)
datatype 'a tree
  = Leaf
  | Branch "'a tree" "'a" "'a tree"
    
term "Branch"
    
fun mem :: "'a \<Rightarrow> 'a tree \<Rightarrow> bool" where
  "mem x Leaf = False" |
  "mem x (Branch l e r) =
     (if x = e then
        True
      else if mem x l then
        True
      else mem x r)"
  
value "mem x Leaf"
  
fun mirror :: "'a tree \<Rightarrow> 'a tree" where
  "mirror Leaf = Leaf" |
  "mirror (Branch l e r) = Branch (mirror r) e (mirror l)"
  
fun as_seq :: "'a tree \<Rightarrow> 'a seq" where
  "as_seq Leaf = \<bullet>" |
  "as_seq (Branch l e r) = app (as_seq l) (app (\<lbrace> e \<rbrace>) (as_seq r))"
  
fun sz :: "'a tree \<Rightarrow> nat" where
  "sz Leaf = 0" |
  "sz (Branch l e r) = 1 + sz l + sz r"
  
lemma mem_mirror [intro]: (*[intro] attribute used to mark introduction rules for automation*)
  assumes "mem x t"
  shows "mem x (mirror t)"
  using assms
  apply(induction t)
   apply simp
  apply(simp split!: if_split_asm) (*split!: aggressively use the if_split_asm theorem to split the if*)
  done
    
thm if_split_asm (*for splitting if in assumptions*)
thm if_split (*for splitting if in conclusions*)
    
lemma mirror_invol [simp]:
  shows "mirror (mirror t) = t"
  by(induction t, auto)
    
lemma sz_len [simp]:
  shows "len (as_seq t) = sz t"
  apply(induction t)
   apply(auto simp add: singleton_def)
  done
    
lemma
  shows "sz (mirror t) = sz t"
  apply(induction t)
   apply auto
  done
    
subsection\<open>Mutually recursive types\<close>
  
(*separate the types using the "and" keyword*)
datatype 'a three_tree
  = Leaf   "'a"
  | Split3 "'a two_tree" "'a two_tree" "'a two_tree"
and 'a two_tree
  = Split2 "'a three_tree" "'a three_tree"
  
datatype way = L | C | R
  
(*mutually recursive types require mutually recursive functions*)
(*define the two functions together using the "and" keyword*)
fun navigate :: "way list \<Rightarrow> 'a three_tree \<Rightarrow> 'a option"
    and navigate' :: "way list \<Rightarrow> 'a two_tree \<Rightarrow> 'a option" where
  "navigate [] (Leaf l) = Some l" |
  "navigate (w#ws) (Split3 l c r) =
     (case w of L \<Rightarrow> navigate' ws l | C \<Rightarrow> navigate' ws c | R \<Rightarrow> navigate' ws r)" |
  "navigate _ _ = None" |
  "navigate' (w#ws) (Split2 l r) =
    (case w of L \<Rightarrow> navigate ws l | R \<Rightarrow> navigate ws r | C \<Rightarrow> None)" |
  "navigate' _ _ = None"
  
thm navigate_navigate'.induct
thm navigate.simps
thm navigate'.simps
  
thm three_tree_two_tree.induct
  
subsection\<open>Records\<close>
  
(*records are non-recursive in Isabelle*)
record processor =
  memory :: "nat \<Rightarrow> int option"
  acc :: "int"
  program_count :: "int"
  program :: "int list"
  
print_theorems

(*record update syntax*)  
term "(p::processor)\<lparr>memory := Map.empty\<rparr>"
(*fields are projected out like so*)
term "acc (p::processor)"
(*combining update and field projection*)
term "p\<lparr>program_count := program_count p + 3\<rparr>"
    
subsection\<open>More in depth example: Tries\<close>

(*this will be useful below*)
term "(\<lambda>x. None) :: 'a \<Rightarrow> 'b option"
term "(\<lambda>x. None) :: 'a \<rightharpoonup> 'b"
term "Map.empty"
  
term "(\<lambda>x. if x = True then Some (0::nat) else None) :: bool \<rightharpoonup> nat"
term "Map.empty(True \<mapsto> (0::nat))" (*function update*)
term "Map.map_of [(True, (0::nat))]"
  
(*tries as a recursive datatype*)
datatype ('k, 'v) trie
  = Trie "'v option" "'k \<rightharpoonup> ('k, 'v) trie"
  
definition empty :: "('k, 'v) trie" where
  "empty \<equiv> Trie None Map.empty"
  
fun lookup :: "'k seq \<Rightarrow> ('k, 'v) trie \<rightharpoonup> 'v" where
  "lookup \<bullet> (Trie v child) = v" |
  "lookup (k \<triangleright> ks) (Trie v child) =
     (case child k of
        None \<Rightarrow> None
      | Some c \<Rightarrow> lookup ks c)"
  
fun insert :: "'k seq \<Rightarrow> 'v \<Rightarrow> ('k, 'v) trie \<Rightarrow> ('k, 'v) trie" where
  "insert \<bullet> v (Trie _ child) = Trie (Some v) child" |
  "insert (k \<triangleright> ks) v1 (Trie v2 child) =
     (case child k of
        None \<Rightarrow> Trie v2 (child(k \<mapsto> insert ks v1 empty))
      | Some c \<Rightarrow> Trie v2 (child(k \<mapsto> insert ks v1 c)))"
  
lemma
  shows "lookup ks (insert ks v t) = Some v"
  apply(induction ks)
   apply(case_tac t, simp)
  apply(case_tac t, simp split!: option.split)
    (*problem: IH not applicable!*)
  oops
  
lemma lookup_insert [simp]:
  shows "lookup ks (insert ks v t) = Some v"
  apply(induction ks arbitrary: t) (*allow t to vary in the inductive hypothesis*)
   apply(case_tac t, clarify)
   apply simp
  apply(case_tac t, clarify)
  apply(simp split!: option.split)
  done
    
lemma insert_insert_overwrite [simp]:
  assumes "ks1 = ks2"
  shows "insert ks1 v1 (insert ks2 v2 t) = insert ks1 v1 t"
using assms
  apply(induction ks1 arbitrary: t ks2) (*allow t and ks2 to vary in the inductive hypothesis*)
   apply(case_tac t, clarify)
   apply simp
  apply(case_tac t, clarify)
  apply(simp split!: option.split)
  done
    
lemma insert_insert_permute:
  assumes "ks1 \<noteq> ks2"
  shows "insert ks1 v1 (insert ks2 v2 t) = insert ks2 v2 (insert ks1 v1 t)"
using assms
  apply(induction ks1 arbitrary: t ks2)
  apply(case_tac t, clarify)
  apply(case_tac ks2)
  apply simp
  apply(simp split!: option.split)
  apply(case_tac t, clarify)
  apply(case_tac ks2)
  apply(auto split!: option.split)
  done
    
fun move_to :: "'k seq \<Rightarrow> ('k, 'v) trie \<rightharpoonup> ('k, 'v) trie" where
  "move_to \<bullet> t = Some t" |
  "move_to (k \<triangleright> ks) (Trie v child) =
     (case child k of
        None \<Rightarrow> None
      | Some c \<Rightarrow> move_to ks c)"
  
lemma navigate_app [simp]:
  shows "move_to (app ks ls) t =
           (case move_to ks t of
              None \<Rightarrow> None
            | Some t \<Rightarrow> move_to ls t)" (*could have split this into two lemmas*)
  apply(induction rule: move_to.induct)
    apply simp
  apply(simp split!: option.split)
  done
    
end
