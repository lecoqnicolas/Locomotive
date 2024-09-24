import git


def get_git_commit():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha
