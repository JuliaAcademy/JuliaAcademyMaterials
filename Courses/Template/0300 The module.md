<!-- https://www.youtube.com/watch?v=dQw4w9WgXcQ -->

# The module

A course will have many modules. Each module can be thought of as one "lecture"
and can have up to four different kinds of resources for that lecture:

* A video
* Markdown text that becomes embedded HTML directly on the lecture page
* A Jupyter notebook for interactive exploration on JuliaBox
* A Jupyter notebook for exercises that the student must complete on JuliaBox

## The video and markdown text

Videos are crucially important for engaging with students and encouraging them
to complete all the materials, and as such they're the banner feature at
JuliaAcademy. They get embedded directly into the lecture page itself, front
and center. Similarly, custom text can also be placed directly into the
JuliaAcademy page underneath the video. Either a video or custom text (or both)
is required for each module in your course.

Videos are large files and cumbersome to work in git repositories, so we use
the simple convention that both the video and markdown text are both specified
within a `.md` file, where the name of that file (without any leading numbers
and without the extension) becomes the name of the module. If the very first
line of that markdown file is a commented link, then that link is taken to be
the location of the video.

For example, a module with _just_ a video and no other content would simply
have one markdown file with the content:

```md
<!-- https://www.youtube.com/watch?v=dQw4w9WgXcQ -->
```

Conversely, a module with _just_ custom embedded HTML would simply start with:

```md
# Lorem ipsum dolor

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
```

As you would expect, we can specify both a video and custom text simply with a
markdown file that has a commented link on its first line and then continues
with content:

```md
<!-- https://www.youtube.com/watch?v=dQw4w9WgXcQ -->
# Lorem ipsum dolor

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
```

## The Jupyter notebooks

We encourage authors to lean heavily on interactive notebooks. Just name your
notebook the same as the module name, just with a `.jl` suffix instead.

Similarly, graded exercises are avaiable, just add the suffix `-quiz.jl`.

See the notebooks from this module for more details.
