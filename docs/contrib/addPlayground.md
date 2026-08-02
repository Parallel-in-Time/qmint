# Add a playground

📜 _To add experimental scripts or usage examples, without testing everything : simply add your own **playground** in {py:mod}`qmint.play`._

Collecting playgrounds on different topics is the main motivation behind `qmint`.
Those are **non-tested but documented** code and scripts that are put on the repository so anyone can play and eventually extend those.
To add your own playground, follow those steps $:$

1. create a folder with a _short & representative_ name, _e.g_ `yoodlidoo` (can also be your name for a personal playground),
2. put your script(s) in it, and document them as much as necessary so **anyone else can understand and use your code**,
3. create a `__init__.py` file in your playground folder with a short summary of your scripts in its docstring, _e.g_
    ```python
    """
    - :class:`script1` : trying some stuff.
    - :class:`script2` : yet another idea.
    """
    ```
4. add the item line corresponding to your playground in `qmint.play.__init__.py`, _e.g_
    ```python
    """
    ...

    Current playgrounds
    -------------------

    - ...
    - :class:`yoodlidoo` : some ideas to do stuff
    """
    ```
5. open a pull request against the `main` branch of `qmint`.

> 💡 If you don't want your playground to be integrated into the main branch of`qmint` (no proper documentation, code always evolving, ...),
> you can still add a **soft link to a playground in your fork** by modifying `qmint.play.__init__.py` :
> ```python
> """
> ...
>
> Current playgrounds
> -------------------
>
> - ...
> - `{name} <https://github.com/{userName}/qmint/tree/{branch}/qmint/play/{name}>`_ : quick description ...
> """
> ```
> where `name` is your playground name, `userName` your GitHub username and `branch` the branch name on your fork you are working on
> (**do not use the `main` branch of your fork** ⚠️)