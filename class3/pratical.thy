theory pratical
  imports Main
begin

(*use rule, erule, and so on combined with basic elimination and introduction rules.  do not use
  any automation...*)
  
lemma
  shows "(\<forall>x. (P \<longrightarrow> Q x)) \<longleftrightarrow> (P \<longrightarrow> (\<forall>x. Q x))"
  apply(rule iffI)
   apply(rule impI)
   apply(rule allI)
   apply(erule_tac x=x in allE)
   apply(erule impE, assumption+)
  apply(rule allI, rule impI)
  apply(erule impE)
   apply assumption
  apply(erule_tac x=x in allE)
    apply assumption
  
(*prove contrapos_np first, then use it to prove the next lemma below:*)
lemma contrapos_np:
  assumes "\<not> Q" and "\<not> P \<longrightarrow> Q"
  shows "P"
  using assms apply -
  thm ccontr
    thm notE
  apply(rule ccontr)
  apply(erule notE)
    apply(erule impE; assumption)
      
    
lemma
  shows "\<not> (\<forall>x. P x \<and> Q x) \<longleftrightarrow> (\<exists>x. \<not> P x \<or> \<not> Q x)"
  apply(rule iffI, rule contrapos_np[where Q="\<forall>x. P x \<and> Q x" and P="(\<exists>x. \<not> P x \<or> \<not> Q x)"], assumption)
   apply(rule allI, rule conjI)
    apply(rule_tac P="P x" in contrapos_np[where Q="(\<exists>x. \<not> P x \<or> \<not> Q x)"], assumption, rule_tac x=x in exI, rule disjI1, assumption)
   apply(rule_tac P="Q x" in contrapos_np[where Q="(\<exists>x. \<not> P x \<or> \<not> Q x)"], assumption, rule_tac x=x in exI, rule disjI2, assumption)
  apply(rule notI, erule exE, erule_tac x=x in allE)
  apply(erule disjE)
   apply(erule conjE, erule notE, assumption)
    apply(erule conjE, erule notE, assumption)
  
lemma
  shows "\<forall>P. \<forall>Q. \<forall>x. ((P x \<longrightarrow> Q x) \<longrightarrow> P x) \<longrightarrow> P x"
  apply(rule allI, rule allI, rule allI)
  apply(case_tac "P x", rule impI, assumption)
  apply(rule impI, erule impE, rule impI, erule notE, assumption)
    apply(assumption)
  done
lemma
  assumes "Q \<longrightarrow> (\<forall>x. S x)"
    and "S False \<longrightarrow> Q"
    and "S True \<longrightarrow> Q"
  shows "Q \<longleftrightarrow> (\<forall>x. S x)"   
  using assms apply -
  apply(rule iffI, erule impE, assumption+)
  apply(erule allE[where x=x])
  apply(case_tac x)
   apply(erule impE) back back
    apply(simp)
   apply(assumption)
  apply(erule impE) back
  apply(simp)
    apply(assumption)
  
lemma
  shows "(\<forall>P Q. \<exists>!x. P x \<or> Q x) \<longrightarrow> False"
    
lemma
  shows "\<exists>P. \<forall>x. \<not> P x"
                  
lemma
  shows "(\<forall>x. P x \<longrightarrow> Q x) \<longleftrightarrow> (\<forall>x. \<not> P x \<or> Q x)"
  apply(rule iffI)
   apply(rule allI, erule_tac x=x in allE)
  apply(case_tac "P x")
    apply(erule impE, assumption, rule disjI2, assumption)
   apply(rule disjI1, assumption)
  apply(rule allI, erule_tac x=x in allE)
    apply(rule impI, erule disjE, erule notE, assumption+)
    
end