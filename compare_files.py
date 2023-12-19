def compare_files(fileA_path, fileB_path):
    # ファイルAのユーザIDを読み込むのだ
    with open(fileA_path, 'r') as fileA:
        usersA = set(line.strip() for line in fileA.readlines())
        
    # ファイルBのユーザIDを読み込むのだ
    with open(fileB_path, 'r') as fileB:
        usersB = set(line.strip() for line in fileB.readlines())

    # ファイルAにあって、ファイルBにないユーザIDを求めるのだ
    only_in_A = usersA - usersB
    
    return only_in_A

if __name__ == "__main__":
    fileA="tweet_user_list/2020-04-76.txt"
    fileB="retweet_user_list_else.txt"
    only_in_A = compare_files(fileA, fileB)
    print()
    for user_id in only_in_A:
        print(user_id)

# qcmd "python compare_files.py >> retweet_user_list_else.txt"