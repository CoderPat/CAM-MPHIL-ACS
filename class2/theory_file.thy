theory theory_file
  imports Main
begin

section\<open>The rules of the (propositional) game\<close>
  
subsection\<open>True and False\<close>
  
term True
term False
  
(*naming scheme: I = intro, E = elim, D = dest *)
thm TrueI
thm FalseE
  
(*note how solve_direct spots this is an immediate consequence of already established facts*)
lemma
  shows "True"
  apply(rule TrueI) (*rule: applies an inference rule backwards*)
  done
    
(*rule: resolve theorem head against conclusion of goal and open a new goal for each of the
  assumptions of the theorem.  key process: higher-order unification*)
    
lemma
  assumes "False"
  shows "A"
  apply(rule FalseE)
  apply(rule assms) (*assms, implicit name of the assumptions*)
  done
    
lemma
  assumes 1: "False" (*note explicitly named assumption (yes, numbers are names, too!)*)
  shows "A"
  by(rule FalseE, rule 1)
    
subsection\<open>Conjunction\<close>
    
(*fat implication arrow is meta-level implication, i.e. English language's "if...then" construct
  when discussing semantics of inference rules in Natural Deduction *)
thm conjI
thm conjunct1
thm conjunct2
thm conjE
  
lemma
  assumes 1: "A" and 2: "B" (*can use "and" to separate assumptions*)
  shows "A \<and> B"
  by(rule conjI; simp add: 1 2)
  
lemma
  shows "A \<and> A \<Longrightarrow> A" (*can use explicit "\<Longrightarrow>" implication instead of "assumes ... shows ..." *)
  apply(erule conjE) (*erule: good for elimination rules*)
    apply assumption (*assumption: solve goal outright by appeal to an assumption*)
  done
    
(*erule:
    1. resolve major premiss of theorem against an assumption
    2. resolve conclusion of theorem against goal
  deletes the assumption that was used in step 1

  again, higher-order unification used*)
    
lemma
  shows "A \<and> A \<Longrightarrow> A"
    by(erule conjE, assumption)
  
lemma
  shows "A \<and> (B \<and> C) \<Longrightarrow> (A \<and> B) \<and> C"
  apply(rule conjI)
   apply(rule conjI)
  apply(erule conjE)
    apply assumption
  apply(erule conjE)
   apply(erule conjE)
   apply assumption
  apply(erule conjE)
   apply(erule conjE)
   apply assumption
  done
    
lemma
  shows "A \<and> (B \<and> C) \<Longrightarrow> (A \<and> B) \<and> C"
  apply(rule conjI, rule conjI) (*"foo, goo": apply foo and then goo to first new subgoal*)
    apply(erule conjE, assumption)
   apply(erule conjE, erule conjE, assumption)+ (*"foo+": apply foo multiple times until failure*)
  done

subsection\<open>Implication\<close>
  
(*note the meta- and object-level implications...*)
thm impI
thm impE
  
lemma
  shows "A \<longrightarrow> A"
  apply(rule impI)
  apply assumption
  done
    
lemma
  shows "A \<longrightarrow> (B \<longrightarrow> A)"
  apply(rule impI)
  apply(rule impI)
  apply assumption
  done
    
lemma
  shows "P \<longrightarrow> (Q \<longrightarrow> P)"
  by((rule impI)+, assumption)
    
lemma
  shows "(P \<longrightarrow> Q) \<Longrightarrow> (Q \<longrightarrow> R) \<Longrightarrow> (P \<longrightarrow> R)"
  apply(rule impI)
  apply(erule impE)
    apply assumption
  apply(erule impE)
   apply assumption
  apply assumption
  done
    
lemma
  shows "P \<and> Q \<longrightarrow> True \<and> P"
  apply(rule impI)
  apply(rule conjI)
   apply(rule TrueI)
  apply(erule conjE)
  apply assumption
  done
    
lemma
  shows "(B \<longrightarrow> False) \<Longrightarrow> (A \<longrightarrow> B) \<Longrightarrow> A \<Longrightarrow> C"
  apply(erule impE) back (*back: explicit backtracking*)
   apply assumption
  apply(erule impE)
   apply assumption
  apply(erule FalseE)
  done

thm impE
    
(*note: erule impE could apply to either
    1. B \<longrightarrow> False
    2. A \<longrightarrow> B
  Isabelle chose (1), the "back" command told Isabelle to choose again, picking (2)
*)
    
lemma
  shows "(B \<longrightarrow> False) \<Longrightarrow> (A \<longrightarrow> B) \<Longrightarrow> A \<Longrightarrow> C"
  by((erule impE, assumption)+, erule FalseE) (*by sequencing with "," backtracking is implicit*)
    
subsection\<open>Disjunction\<close>
  
thm disjI1
thm disjI2
thm disjE
  
lemma
  shows "False \<or> Q \<longrightarrow> Q"
  apply(rule impI)
  apply(erule disjE)
   apply(erule FalseE)
  apply assumption
  done
    
lemma
  shows "P \<or> Q \<longrightarrow> Q \<or> P"
  apply(rule impI)
  apply(erule disjE)
   apply(rule disjI2, assumption)
  apply(rule disjI1, assumption)
  done
    
subsection\<open>Negation\<close>
  
thm notE
thm notI
  
(*yes, we really are working in classical logic*)
thm ccontr
thm classical
  
thm excluded_middle
  
lemma conj_disj_demorgan1:
  shows "\<not> P \<or> \<not> Q \<longrightarrow> \<not> (P \<and> Q)"
  apply(rule impI)
  apply(rule notI)
  apply(erule conjE)
  apply(erule disjE)
   apply(erule notE, assumption)+
  done
  
lemma
  shows "\<not> (P \<and> Q) \<longrightarrow> \<not> P \<or> \<not> Q"
  apply(rule impI)
  apply(insert excluded_middle[where P=P]) (*insert: weakens assumptions with a fact*)
  apply(insert excluded_middle[where P=Q]) (*[where x=foo] attribute: explicitly instantiates metavariable in theorem*)
  apply(erule disjE)
   apply(rule disjI1, assumption)
  apply(erule disjE)
   apply(rule disjI2, assumption)
  apply(erule notE)
  apply(rule conjI, assumption, assumption)
  done
    
lemma conj_disj_demorgan2:
  shows "\<not> (P \<and> Q) \<longrightarrow> \<not> P \<or> \<not> Q"
  apply(rule impI)
  apply(case_tac "P"; case_tac "Q") (*case_tac: perform case analysis on "P" and "Q"*)
     apply(erule notE, rule conjI; assumption)
    apply(rule disjI2, assumption)
   apply(rule disjI1, assumption)+
  done
    
(*Peirce's Law*)
lemma
  shows "((P \<longrightarrow> Q) \<longrightarrow> P) \<longrightarrow> P"
  apply(case_tac P)
   apply(rule impI, assumption)
  apply(rule impI)
  apply(erule impE)
   apply(rule impI)
   apply(erule notE, assumption)
  apply assumption
  done
    
subsection\<open>Equality is bi-implication, at type bool\<close>
  
(*bi-implication specific*)
thm iffI
thm iffE
  
(*equality is an equivalence relation*)
thm refl
thm sym
thm trans
  
(*equality is also a congruence relation (the smallest such congruence relation)*)
thm subst
thm cong
  
(*should be able to solve this with conj_disj_demorgan1 and conj_disj_demorgan2, but need to massage
  object-level implication "\<longrightarrow>" into meta-level implication "\<Longrightarrow>"*)
lemma
  shows "(\<not> (P \<and> Q)) = (\<not> P \<or> \<not> Q)"
  oops
    
thm conj_disj_demorgan1
thm conj_disj_demorgan2
thm conj_disj_demorgan1[rule_format] (*[rule_format] attribute: convert object- to meta-implication*)
thm conj_disj_demorgan2[rule_format]
  
lemma
  shows "\<not> (P \<and> Q) \<longleftrightarrow> \<not> P \<or> \<not> Q"
  apply(rule iffI)
   apply(rule conj_disj_demorgan2[rule_format], assumption)
  apply(rule conj_disj_demorgan1[rule_format], assumption)
  done

lemma
  shows "(P \<longrightarrow> Q) = (\<not> Q \<longrightarrow> \<not> P)"
  apply(rule iffI)
   apply(rule impI)
   apply(rule notI)
   apply(erule notE)
   apply(erule impE, assumption, assumption)
  apply(rule impI)
  apply(case_tac "Q", assumption)
  apply(erule impE, assumption)
  apply(erule notE) back
  apply assumption
  done
    
thm arg_cong
    
lemma
  assumes "s = t" and "t = u"
  shows "P s = P u"
  using assms
  apply -
  apply(rule trans[where s="P t"])
   apply(rule arg_cong[where f="P"])
   apply assumption
  apply(rule arg_cong[where f="P"])
  apply assumption
  done

end