theory Rules_Of_The_Predicate_Game
  imports Main
begin

section\<open>The (predicate) rules of the game\<close>
  
subsection\<open>Universal quantification\<close>
  
(*note meta- and object-level universal quantifiers*)
thm allI
thm allE
  
lemma
  assumes "\<forall>x. P \<longrightarrow> Q x"
  shows "P \<longrightarrow> (\<forall>x. Q x)"
  using assms
    apply -
  apply(rule impI)
  apply(rule allI)
  apply(erule_tac x=x in allE) (*erule_tac: permits metavariable instantiation with bound variables*)
  apply(erule impE, assumption)
  apply assumption
  done
    
lemma
  assumes "\<forall>x. P \<longrightarrow> Q x"
  shows "P \<longrightarrow> (\<forall>x. Q x)"
  using assms
    apply -
  apply(rule impI)
  apply(rule allI)
  apply(erule allE) (*don't actually need the instantiation above, though!*)
  apply(erule impE, assumption)
  apply assumption
  done
    
lemma
  shows "(\<forall>x. P x \<and> Q x) \<longleftrightarrow> (\<forall>x. P x) \<and> (\<forall>x. Q x)"
  apply(rule iffI)
   apply(rule conjI)
    apply(rule allI)
    apply(erule_tac x=x in allE)
    apply(erule conjE, assumption)
    apply(rule allI)
   apply(erule_tac x=x in allE)
   apply(erule conjE, assumption)
  apply(erule conjE)
  apply(rule allI)
  apply(erule_tac x=x in allE)+
  apply(rule conjI; assumption)
  done
    
lemma
  assumes "\<forall>x. P x \<or> Q x \<longrightarrow> R"
    and "P y"
  shows "R"
  using assms
  apply -
  apply(erule allE[where x=y]) (*note "y" is not locally bound, here, so don't need erule_tac*)
  apply(erule impE)
   apply(rule disjI1, assumption)
  apply assumption
  done
    
subsection\<open>Existential quantification\<close>
  
thm exI
thm exE
  
lemma
  shows "(\<exists>x. P x \<or> Q x) \<longleftrightarrow> (\<exists>x. P x) \<or> (\<exists>x. Q x)"
  apply(rule iffI)
   apply(erule exE)
   apply(erule disjE)
    apply(rule disjI1)
    apply(rule exI)
    apply assumption
   apply(rule disjI2)
   apply(rule exI)
   apply assumption
  apply(erule disjE)
   apply(erule exE)
   apply(rule exI)
   apply(rule disjI1)
   apply assumption
  apply(erule exE)
  apply(rule exI)
  apply(rule disjI2)
  apply assumption
  done
    
lemma
  shows "(\<exists>x. P \<or> Q x) \<longleftrightarrow> P \<or> (\<exists>x. Q x)"
  apply(rule iffI)
   apply(erule exE)
   apply(erule disjE)
    apply(rule disjI1)
    apply assumption
   apply(rule disjI2)
  apply(rule exI)
   apply assumption
  apply(erule disjE)
   apply(rule exI)
   apply(rule disjI1)
   apply assumption
  apply(erule exE)
  apply(rule exI)
  apply(rule disjI2)
  apply assumption
  done
    
subsection\<open>Unique existence\<close>
  
thm ex1I
thm ex1E
  
lemma
  assumes "\<exists>!x. P x"
  shows "\<exists>x. P x"
  using assms
    apply -
  apply(erule ex1E)
  apply(rule exI)
  apply assumption
  done
  
end