#!/bin/bash

# コミットメッセージを入力するのだ
echo "コミットメッセージを入力するのだ:"
read commit_message

# リモートリポジトリとブランチを指定するのだ
remote_repository="origin"
branch="main"

# add, commit, そして push のコマンドを実行するのだ
git add *.py *.sh *.png *.pdf *.csv *.txt
git commit -m "$commit_message"
git push $remote_repository $branch

echo "プッシュ完了なのだ！"
