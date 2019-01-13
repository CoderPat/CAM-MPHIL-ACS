theory theory_file
  imports Main
begin

section\<open>Typed set theory\<close>

(*some finite sets*)
term "{}"
term "{0,1,2,(3::nat)}"
term "{True, False}"

(*a simple set comprehension*)
term "{x. x \<le> (5::nat) }"
  
(*a set defined by a set comprehension, and computing with that comprehension*)
term "{ x. x \<in> {True,False} \<and> x \<longrightarrow> \<not> x }"
value "{ x. x \<in> {True,False} \<and> x \<longrightarrow> \<not> x }"
  
(*the universal set (at a given type)*)
term "UNIV"
  
(*some familiar relations on sets*)
term "P \<union> Q"
term "P \<inter> Q"
term "x \<in> P"
term "P \<subseteq> Q"
term "(P::'a set) - Q"
term "P = (Q::'a set)"
  
(*note some relations also have abbreviated negated forms*)
term "x \<notin> P"
term "P \<noteq> Q"
  
(*an infinite set, and computing with that infinite set*)
term "UNIV - {1,2,3::nat}"
value "UNIV - {1,2,3::nat}"
  
subsection\<open>Intersection and union\<close>
  
thm IntI
thm IntE
thm IntD1
thm IntD2

thm UnE
thm UnI1
thm UnI2
  
(*A faulty lemma: but quickcheck to the rescue!*)
lemma
  assumes "x \<in> A \<inter> (B \<union> C)"
  shows "x \<in> A \<inter> C"
  quickcheck (*tries to find counterexamples to theorem statements*)
  oops
    
(*quickcheck works by either
    1. exhaustively enumerating all possible values for the variables in the statement
    2. randomly generating values for variables in the statement
  (you can configure which mode it uses manually, but by default it uses exhaustive checking)

  "nitpick" is another automated tool for refuting theorem statements which uses SAT technology*)
    
lemma
  assumes "x \<in> A \<inter> (B \<union> C)"
  shows "x \<in> A \<union> C"
  using assms
  apply -
  apply(erule IntE)
  apply(rule UnI1)
  apply assumption
  done
    
lemma
  shows "x \<in> P \<union> Q \<longleftrightarrow> x \<in> P \<or> x \<in> Q"
  apply(rule iffI)
   apply(erule UnE)
    apply(rule disjI1, assumption)
   apply(rule disjI2, assumption)
  apply(erule disjE)
  apply(rule UnI1, assumption)
  apply(rule UnI2, assumption)
  done
    
lemma
  shows "x \<in> P \<inter> Q \<longleftrightarrow> x \<in> P \<and> x \<in> Q"
  apply(rule iffI)
   apply(erule IntE)
   apply(rule conjI; assumption) (*foo;goo = first apply foo then apply goo to all new goals*)
  apply(erule conjE)
  apply(rule IntI; assumption)
  done
    
subsection\<open>Subset relations\<close>
  
(*improper relations*)
thm subsetI
thm subsetD
  
(*proper relations*)
thm psubsetI
thm psubsetE
thm psubsetD
  
lemma
  assumes "P \<subseteq> R" and "Q \<subseteq> R"
  shows "P \<union> Q \<subseteq> R"
  using assms
  apply -
  apply(rule subsetI)
  apply(erule UnE)
   apply(erule subsetD, assumption)+
  done
    
lemma
  assumes "P \<subseteq> Q" and "Q \<subseteq> R"
  shows "P \<subseteq> R"
  using assms
    apply -
  apply(rule subsetI)
  apply(erule subsetD)
  apply(erule subsetD)
  apply assumption
  done
    
lemma
  assumes "P \<subset> Q"
    and "x \<notin> Q"
  shows "x \<notin> P"
  using assms
  apply -
  apply(rule notI)
  apply(drule psubsetD, assumption) (*drule used for forward reasoning from assumptions*)
  apply(erule notE, assumption)
  done
    
(* drule
     1. resolves major premiss of theorem against an assumption
     2. deletes that assumption
     3. inserts conclusion of theorem as new assumption
     4. opens new subgoals for remaining assumptions of theorem
   frule is same, apart from the deletion step in (2)
*) 
    
subsection\<open>Set equality\<close>
  
(*equality at set type is mutual subset inclusion*)
thm equalityI
thm equalityE
  
lemma
  shows "A \<inter> (B \<union> C) = (A \<inter> B) \<union> (A \<inter> C)"
  apply(rule equalityI)
   apply(rule subsetI)
   apply(erule IntE)
   apply(erule UnE)
    apply(rule UnI1)
    apply(rule IntI)
     apply assumption+
   apply(rule UnI2)
   apply(rule IntI)
    apply assumption+
  apply(rule subsetI)
  apply(erule UnE)
   apply(erule IntE)
   apply(rule IntI)
    apply assumption
   apply(rule UnI1)
   apply assumption
  apply(erule IntE)
  apply(rule IntI)
   apply assumption
  apply(rule UnI2)
  apply assumption
  done
    
lemma
  fixes A B :: "'a set" (*fixes keyword: used to provide optional type annotation for variables*)
  assumes "A = B"
    and "x \<notin> A" (*recall x\<notin>A is abbreviation for \<not>(x \<in> A)*)
  shows "x \<notin> B"
  using assms
  apply -
  apply(rule notI)
  apply(erule equalityE)
  apply(drule subsetD) back
    apply(assumption)
  apply(erule notE)
  apply assumption
  done
    
lemma
  assumes "A \<subseteq> B"
  shows "A \<inter> B = A"
  using assms
  apply -
  apply(rule equalityI; rule subsetI)
   apply(erule IntE, assumption)
  apply(rule IntI)
   apply assumption
  apply(erule subsetD, assumption)
  done
    
subsection\<open>Empty and universal sets\<close>
  
thm UNIV_I
thm emptyE
    
lemma
  shows "A \<subseteq> UNIV"
  apply(rule subsetI)
  apply(rule UNIV_I)
  done
    
lemma
  assumes "x \<in> P \<union> {}"
  shows "x \<in> P"
  using assms
  apply -
  apply(erule UnE)
   apply assumption
  apply(erule emptyE)
  done
    
lemma
  shows "P \<inter> {} = {}"
  apply(rule equalityI; rule subsetI)
   apply(erule IntE)
   apply assumption
  apply(erule emptyE)
  done
    
subsection\<open>Set difference\<close>
  
thm DiffI
thm DiffD1
thm DiffD2
thm DiffE
  
lemma
  shows "A - (B \<union> C) = (A - B) - C"
  apply(rule equalityI; rule subsetI)
   apply(erule DiffE)
   apply(rule DiffI)
    apply(rule DiffI)
     apply assumption
    apply(rule notI)
    apply(erule notE)
    apply(rule UnI1)
    apply assumption
   apply(rule notI)
   apply(erule notE)
   apply(rule UnI2)
   apply assumption
  apply(erule DiffE)
  apply(erule DiffE)
  apply(rule DiffI)
   apply assumption
  apply(rule notI)
  apply(erule UnE)
   apply(erule notE, assumption)+
  done
    
lemma
  shows "A - A = {}"
  apply(rule equalityI; rule subsetI)
   apply(erule DiffE)
   apply(erule notE, assumption)
  apply(erule emptyE)
  done
    
subsection\<open>Set comprehensions\<close>
term Collect
  
thm CollectD
thm CollectI
  
lemma
  assumes "x \<in> {y. P y}" and "x \<in> {y. Q y}"
  shows "x \<in> {y. P y \<and> Q y}"
  using assms
  apply -
  apply(drule CollectD)
  apply(drule CollectD)
  apply(rule CollectI)
  apply(rule conjI)
   apply assumption+
  done
    
lemma
  shows "{x. P x} \<subseteq> {x. Q x} \<longleftrightarrow> (\<forall>x. P x \<longrightarrow> Q x)"
  apply(rule iffI)
   apply(rule allI)
   apply(rule impI)
   apply(drule subsetD, rule CollectI, assumption)
   apply(drule CollectD)
   apply assumption
  apply(rule subsetI)
  apply(drule CollectD)
  apply(erule allE)
  apply(erule impE)
   apply assumption
  apply(rule CollectI)
  apply assumption
  done
    
lemma
  shows "{} = {x. False}"
  apply(rule equalityI; rule subsetI)
   apply(erule emptyE)
  apply(drule CollectD)
  apply(erule FalseE)
  done
    
subsection\<open>Bounded quantifiers\<close>
  
term "\<forall>x\<in>S. P"
term "\<exists>x\<in>S. P"
  
thm ballI
thm ballE
  
thm bexI
thm bexE
  
lemma
  assumes "\<forall>x\<in>S. P" and "\<forall>x\<in>T. P"
  shows "\<forall>x\<in>S\<union>T. P"
  using assms
  apply -
  apply(rule ballI)
  apply(erule UnE)
  apply(erule_tac x=x in ballE)
    apply assumption
   apply(erule notE, assumption)
  apply(erule_tac x=x in ballE) back
   apply assumption
  apply(erule notE, assumption)
  done
    
lemma
  assumes "\<exists>x\<in>{}. P"
  shows "Q"
  using assms
    apply -
  apply(erule bexE)
  apply(erule emptyE)
  done
    
lemma
  shows "(\<forall>x\<in>UNIV. P x) \<longleftrightarrow> (\<forall>x. P x)"
  apply(rule iffI)
   apply(rule allI)
   apply(erule_tac x=x in ballE)
    apply assumption
   apply(erule notE)
   apply(rule UNIV_I)
  apply(rule ballI)
  apply(erule_tac x=x in allE)
  apply assumption
  done
  
subsection\<open>Image\<close>
  
(*set image is a map-like function*)
term "f ` S"
term "image"
  
value "((op +) 1) ` {(0::nat), 1, 2, 3}"

thm imageI
thm imageE
  
lemma
  assumes "x \<in> f ` S" and "x \<in> f ` T"
  shows "x \<in> f ` (S \<union> T)"
  using assms
  apply -
  apply(erule imageE)
  apply(erule imageE)
  apply clarify (*clarify: "clarifies" goal by rewriting throughout with assumed equalities*)
  apply(rule imageI)
  apply(rule UnI1)
  apply assumption
  done
    
section\<open>"Big" union and intersection\<close>
  
term "\<Union>i\<in>S. f i"
term "\<Inter>i\<in>S. f i"
  
value "\<Union>i\<in>{True,False}. {i \<and> False}" (*bounded form*)
value "\<Inter>{{1,2,(3::nat)},{2,3},{2}}"  (*unbounded form*)
  
thm UNION_eq
thm INTER_eq
  
lemma
  shows "(\<Inter>i\<in>S. f i) \<inter> (\<Inter>i\<in>S. g i) = (\<Inter>i\<in>S. f i \<inter> g i)"
  apply(rule equalityI; rule subsetI)
   apply(erule IntE)
   apply(subst INTER_eq) (*subst: substitutes using an equational theorem in the conclusion of the goal*)
   apply(subst (asm) INTER_eq) (* subst (asm): substitutes using an equational theorem in the assumption of the goal*)
   apply(subst (asm) INTER_eq)
   apply(drule CollectD)+
   apply(rule CollectI)
   apply(rule ballI)
   apply(rename_tac xa) (*rename_tac: renames bound variable (Isabelle auto-generated names are not stable)*)
   apply(erule_tac x=xa in ballE)+
     apply(rule IntI; assumption)
    apply(erule notE, assumption)+
  apply(subst INTER_eq)+
  apply(subst (asm) INTER_eq)
  apply(rule IntI)
   apply(drule CollectD)
   apply(rule CollectI)
   apply(rule ballI)
    apply(rename_tac xa)
   apply(erule_tac x=xa in ballE)
    apply(erule IntE)
    apply assumption
   apply(erule notE)
   apply assumption
  apply(drule CollectD)
  apply(rule CollectI)
  apply(rule ballI)
  apply(rename_tac xa)
  apply(erule_tac x=xa in ballE)
   apply(erule IntE)
   apply assumption
  apply(erule notE)
  apply assumption
  done
    
lemma
  shows "j \<in> S \<Longrightarrow> f j \<subseteq> (\<Union>i\<in>S. f i)"
  apply(rule subsetI)
  apply(subst UNION_eq)
  apply(rule CollectI)
  apply(rule bexI[where x=j])
   apply assumption
    apply assumption
  done
    
lemma
  shows "j \<in> S \<Longrightarrow> f j \<subseteq> (\<Union>i\<in>S. f i)"
  by blast (*tableaux prover good for sets and logical reasoning*)

end