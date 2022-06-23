# pyutils

!pip install git+https://github.com/rfmanz/pyutils

eval "$(ssh-agent -s)"

ssh-add ~/.ssh/id_ed25519


!GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519' pip install git+ssh://git@github.com/rfmanz/pyutils.git
