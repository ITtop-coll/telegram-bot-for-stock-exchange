import subprocess
import sys

print("=" * 50)
print("ИСПРАВЛЕНИЕ ОШИБКИ AIOGRAM")
print("=" * 50)

python_exe = sys.executable.replace("pythonw.exe", "python.exe")

print("\n1. Удаляю старую версию aiogram...")
subprocess.run([python_exe, "-m", "pip", "uninstall", "aiogram", "-y"], capture_output=True)
print("Удалено")

print("\n2. Устанавливаю aiogram 2.25.1...")
result = subprocess.run([python_exe, "-m", "pip", "install", "aiogram==2.25.1"], capture_output=True, text=True)

if result.returncode == 0:
    print("aiogram 2.25.1 установлен")
else:
    print("Ошибка")
    print(result.stderr)

print("\n" + "=" * 50)
print("запускайте telegram_bot.py")
print("=" * 50)
input("\nНажмите Enter для выхода...")