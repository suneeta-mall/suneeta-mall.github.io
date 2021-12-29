FROM jekyll/jekyll:latest

RUN mkdir -p blog
WORKDIR blog
# COPY . .
# RUN chmod 777 blog

RUN  jekyll clean 

ENTRYPOINT ["jekyll", "serve", "--watch"]
