from notify_run import Notify

notify = Notify()


def alert_me(name):
    message = name + " sat down in front of your PC in your absence!!"
    notify.send(message)


def alert_me2(num):
    message = str(num) + " unknown faces detected!!"
    notify.send(message)
