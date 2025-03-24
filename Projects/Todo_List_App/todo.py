import os

TASKS_FILE = "tasks.txt"

def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r") as f:
        tasks = [line.strip() for line in f.readlines()]
    return tasks

def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        for task in tasks:
            f.write(task + "\n")

def display_tasks(tasks):
    if not tasks:
        print("No tasks found.")
        return
    print("\nTasks:")
    for idx, task in enumerate(tasks, start=1):
        print(f"{idx}. {task}")

def main():
    tasks = load_tasks()

    while True:
        print("\nTodo List App")
        print("1. View Tasks")
        print("2. Add Task")
        print("3. Mark Task as Completed")
        print("4. Delete Task")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            display_tasks(tasks)

        elif choice == "2":
            task = input("Enter new task: ")
            tasks.append(task)
            save_tasks(tasks)
            print("Task added.")

        elif choice == "3":
            display_tasks(tasks)
            try:
                num = int(input("Enter task number to mark as completed: "))
                tasks[num-1] += " [Completed]"
                save_tasks(tasks)
                print("Task marked as completed.")
            except (IndexError, ValueError):
                print("Invalid task number.")

        elif choice == "4":
            display_tasks(tasks)
            try:
                num = int(input("Enter task number to delete: "))
                tasks.pop(num-1)
                save_tasks(tasks)
                print("Task deleted.")
            except (IndexError, ValueError):
                print("Invalid task number.")

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
