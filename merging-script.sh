
#!/bin/sh

repositories=(MLADM-17-18 IFV-17-18 PML-17-18 MPHIL-PROJECT)

shopt -s extglob
for repo in ${repositories[@]}
do
	git remote add -f ${repo} git@github.com:CoderPat/${repo}.git
	git merge --allow-unrelated-histories ${repo}/master -m "merge ${repo} into the main one"
	mkdir ${repo} 
	mv !($(IFS="|" ; echo "${repositories[*]}")|merging-script.sh) ${repo}/ 
	git add -A .
	git commit -m "move subrepository ${repo} to its own folder"
done
