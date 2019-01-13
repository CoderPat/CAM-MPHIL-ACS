theory pratical
  imports Main
begin
  
(*use rule, erule, and so on combined with basic elimination and introduction rules.  do not use
  any automation...*)
  
thm conjI
thm conjunct1  
thm impE
  
lemma 
  shows "P \<longrightarrow> P \<and> P"
  apply (rule impI)
  apply (rule conjI, assumption, assumption)
  done
    
lemma
  shows "P \<or> False \<longrightarrow> P"
  apply (rule impI)
  apply (erule disjE)
   apply assumption
  apply(erule FalseE)
 done

lemma 
  shows "P \<or> (Q \<or> R) \<longrightarrow> (P \<or> Q) \<or> R"
  apply (rule impI)
  apply (erule disjE)
   apply(rule disjI1, rule disjI1, assumption)
  apply(erule disjE)
   apply(rule disjI1, rule disjI2, assumption)
    apply(rule disjI2, assumption)
    
lemma
  shows "P \<and> Q \<longrightarrow> P"
  apply(rule impI, rule conjunct1)  
    
lemma
  shows "(P \<and> Q \<longrightarrow> R) \<longleftrightarrow> (P \<longrightarrow> Q \<longrightarrow> R)"
  apply(rule iffI)
    thm conjE
     apply (rule impI, rule impI, erule impE)
      apply(rule conjI, (assumption)+)
    apply(rule impI)
    apply(erule conjE, erule impE, assumption, erule impE, assumption, assumption)
    done
lemma
  assumes 1: "P \<and> (Q \<and> R)"
  shows "Q"
  apply(rule conjunct1, rule conjunct2, rule \<open>P \<and> (Q \<and> R)\<close>)
  done
    
lemma
  assumes 1:"P \<or> Q \<longrightarrow> R"
    and 2:"R \<longleftrightarrow> S"
  shows "P \<longrightarrow> S" 
  using assms apply -
  apply (rule impI, erule iffE, erule impE)
   apply(rule disjI1, assumption)
  apply(erule impE, assumption)
    apply(erule impE, assumption, assumption)
  done
    
lemma
  shows "((P \<longrightarrow> Q) \<longrightarrow> P) \<longrightarrow> P"
  apply(case_tac P)
   apply(rule impI, assumption)
  apply(rule impI, erule impE)
   apply(rule impI, erule notE, assumption)
    apply(erule notE, assumption)
    
lemma
  shows "\<not>\<not>\<not>P \<longrightarrow> \<not>P"
  apply(rule impI, rule notI) 
  apply(erule notE)
  apply (rule notI, erule notE, assumption)
    
lemma
  assumes "P \<longrightarrow> \<not> Q \<or> R"
    and "R \<longrightarrow> \<not> Q"
    and "S \<and> P"
  shows "\<not> Q"
  using assms apply -
  apply(erule impE)
   apply(erule conjunct2)
  apply(erule disjE,assumption)
  apply(erule impE, assumption)
  apply(assumption)
    
lemma
  shows "\<not> False"
  apply(rule notI, assumption)
  
    
lemma
  shows "\<not> (P \<and> Q) \<longleftrightarrow> (\<not>\<not>\<not>P \<or> \<not>Q)"
  apply(rule iffI)
   apply(insert excluded_middle[where P=Q])
   apply(erule disjE)
    apply(rule disjI2, assumption)
   apply(rule disjI1, rule notI, erule notE)
   apply(rule conjI)
    apply(rule ccontr, erule notE, assumption)
   apply(assumption)
    
    
lemma
  shows "(True \<longrightarrow> P) \<longleftrightarrow> P"
  sorry
  
end