# Git Quick Start Guide

It looks like you missed a keyword in a few of your terminal commands! Here is your step-by-step guide to fixing your remote and successfully pushing your new code (PPO, DQN, RPPO) to GitHub.

## 1. Fix the Remote URL
In your terminal, you tried to run `git remote orgin <URL>`, but you missed the word **`add`**. Also, we standardly use the spelling `origin` (not `orgin`). 

Run this exact command to connect your local code to your GitHub repository:
```bash
git remote add origin https://github.com/sxdxde/Dynamic-Request-Batching-Reinforcement-Learning-Agent.git
```
*(Notice the `.git` at the end of the URL!)*

## 2. The 3-Step "Save and Upload" Process

Every time you want to save your work to GitHub, you **always** have to run these three commands in exactly this order. 

Your `git commit` command failed earlier because you skipped Step 1!

### Step 1: Stage the files (Tell Git what to look at)
You must "add" your changed files to the staging area before you can commit them.
```bash
git add .
```
*(The `.` means "add every file that has changed in this folder")*

### Step 2: Commit (Take a snapshot)
Now that Git knows which files to look at, wrap them up in a commit with a message.
```bash
git commit -m "Added Recurrent PPO and DQN with integration guides"
```

### Step 3: Push (Upload to GitHub)
Finally, send that snapshot to your remote repository on the `sudarshan` branch.
```bash
git push -u origin sudarshan
```
*(If it gives you an error saying the remote has work you don't have, you can force it to overwrite the GitHub version using `git push -u origin sudarshan -f`)*

---

## Summary Cheat Sheet
Whenever you change code and are ready to save to GitHub, you just type:
```bash
git add .
git commit -m "your description of what you changed here"
git push
```
