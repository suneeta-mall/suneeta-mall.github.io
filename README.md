# Blogs and pages
Personal blog and pages powered by [Jekyll](https://github.com/jekyll/jekyll) & [Polar](https://github.com/neizod/polar).

# Run Locally
```bash
jekyll clean    
jekyll serve --watch
```



```
docker build -t blog .
docker run -ti -p 4000:4000  -v `pwd`:`pwd` -w `pwd` --name=blog blog
```