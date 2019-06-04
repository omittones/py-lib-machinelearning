# pylint: skip-file

import clr
clr.AddReference("System.Windows.Forms")
clr.AddReference(r"D:\Code\cs-lib-machine-learning\Environments\bin\Debug\Environments.exe")

from System import *
from System.Threading.Tasks import Task
from Environments import GUI
from Environments.Bouncies import Environment, Renderer
from Environments.Forms import EnvironmentDisplay
from System.Windows.Forms import Form


def main():

    env = Environment()
    env.FrictionFactor = 1

    def create_view():
        form = EnvironmentDisplay()
        renderer = Renderer(env)
        form.Renderer = renderer
        return form

    stop = False
    def loop():
        env.Reset()
        while not stop:
            env.Step(None)

    gui = GUI.ShowForm(Func[Form](create_view))
    engine = Task.Run(Action(loop))
    gui.Wait()
    stop = True
    engine.Wait()

    if gui.Exception != None or engine.Exception != None:
        print(gui.Exception or engine.Exception)

if __name__ == '__main__':
    main()