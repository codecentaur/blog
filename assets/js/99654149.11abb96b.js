"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[870],{1704:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>d,contentTitle:()=>s,default:()=>u,frontMatter:()=>o,metadata:()=>i,toc:()=>l});var r=n(4848),a=n(8453);const o={slug:"adding-learner",title:"Adding Learner",tags:["math-learner"]},s=void 0,i={permalink:"/blog/adding-learner",editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/blog/2024-07-30-adding-learner.md",source:"@site/blog/2024-07-30-adding-learner.md",title:"Adding Learner",description:"In this post, I'm going to describe building a simple neural network using PyTorch that learns to add two numbers.",date:"2024-07-30T00:00:00.000Z",tags:[{inline:!0,label:"math-learner",permalink:"/blog/tags/math-learner"}],readingTime:3.875,hasTruncateMarker:!0,authors:[],frontMatter:{slug:"adding-learner",title:"Adding Learner",tags:["math-learner"]},unlisted:!1,prevItem:{title:"MNIST Autoencoder",permalink:"/blog/MNIST-autoencoder"}},d={authorsImageUrls:[]},l=[{value:"Data Generation",id:"data-generation",level:3}];function c(e){const t={code:"code",h3:"h3",p:"p",pre:"pre",...(0,a.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(t.p,{children:"In this post, I'm going to describe building a simple neural network using PyTorch that learns to add two numbers."}),"\n",(0,r.jsx)(t.h3,{id:"data-generation",children:"Data Generation"}),"\n",(0,r.jsxs)(t.p,{children:["First, I'll define a method that generates the data. This method returns two tensors. The first tensor, ",(0,r.jsx)(t.code,{children:"x"}),", is the input to the model, and the second tensor, ",(0,r.jsx)(t.code,{children:"y"}),", is the expected output (the sum of the pairs of numbers)."]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",children:"def generate_data(num_samples=1000):\n    x = torch.randint(0, 100, (num_samples, 2), dtype=torch.float32)\n    y = torch.sum(x, dim=1, keepdim=True)\n    return x, y\n"})})]})}function u(e={}){const{wrapper:t}={...(0,a.R)(),...e.components};return t?(0,r.jsx)(t,{...e,children:(0,r.jsx)(c,{...e})}):c(e)}},8453:(e,t,n)=>{n.d(t,{R:()=>s,x:()=>i});var r=n(6540);const a={},o=r.createContext(a);function s(e){const t=r.useContext(o);return r.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function i(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:s(e.components),r.createElement(o.Provider,{value:t},e.children)}}}]);