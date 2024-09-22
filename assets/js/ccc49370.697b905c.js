"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[249],{3858:(e,n,t)=>{t.r(n),t.d(n,{default:()=>b});t(6540);var i=t(4164),a=t(1213),s=t(7559),r=t(4096),o=t(8027),l=t(8375),c=t(1312),d=t(9022),u=t(4848);function m(e){const{nextItem:n,prevItem:t}=e;return(0,u.jsxs)("nav",{className:"pagination-nav docusaurus-mt-lg","aria-label":(0,c.T)({id:"theme.blog.post.paginator.navAriaLabel",message:"Blog post page navigation",description:"The ARIA label for the blog posts pagination"}),children:[t&&(0,u.jsx)(d.A,{...t,subLabel:(0,u.jsx)(c.A,{id:"theme.blog.post.paginator.newerPost",description:"The blog post button label to navigate to the newer/previous post",children:"Newer post"})}),n&&(0,u.jsx)(d.A,{...n,subLabel:(0,u.jsx)(c.A,{id:"theme.blog.post.paginator.olderPost",description:"The blog post button label to navigate to the older/next post",children:"Older post"}),isNext:!0})]})}function h(){const{assets:e,metadata:n}=(0,r.e7)(),{title:t,description:i,date:s,tags:o,authors:l,frontMatter:c}=n,{keywords:d}=c,m=e.image??c.image;return(0,u.jsxs)(a.be,{title:t,description:i,keywords:d,image:m,children:[(0,u.jsx)("meta",{property:"og:type",content:"article"}),(0,u.jsx)("meta",{property:"article:published_time",content:s}),l.some((e=>e.url))&&(0,u.jsx)("meta",{property:"article:author",content:l.map((e=>e.url)).filter(Boolean).join(",")}),o.length>0&&(0,u.jsx)("meta",{property:"article:tag",content:o.map((e=>e.label)).join(",")})]})}var f=t(5260);function g(){const e=(0,r.J_)();return(0,u.jsx)(f.A,{children:(0,u.jsx)("script",{type:"application/ld+json",children:JSON.stringify(e)})})}var p=t(3691),x=t(1689);function v(e){let{sidebar:n,children:t}=e;const{metadata:i,toc:a}=(0,r.e7)(),{nextItem:s,prevItem:c,frontMatter:d}=i,{hide_table_of_contents:h,toc_min_heading_level:f,toc_max_heading_level:g}=d;return(0,u.jsxs)(o.A,{sidebar:n,toc:!h&&a.length>0?(0,u.jsx)(p.A,{toc:a,minHeadingLevel:f,maxHeadingLevel:g}):void 0,children:[(0,u.jsx)(x.A,{metadata:i}),(0,u.jsx)(l.A,{children:t}),(s||c)&&(0,u.jsx)(m,{nextItem:s,prevItem:c})]})}function b(e){const n=e.content;return(0,u.jsx)(r.in,{content:e.content,isBlogPostPage:!0,children:(0,u.jsxs)(a.e3,{className:(0,i.A)(s.G.wrapper.blogPages,s.G.page.blogPostPage),children:[(0,u.jsx)(h,{}),(0,u.jsx)(g,{}),(0,u.jsx)(v,{sidebar:e.sidebar,children:(0,u.jsx)(n,{})})]})})}},2234:(e,n,t)=>{t.d(n,{A:()=>c});t(6540);var i=t(4164),a=t(4084),s=t(7559),r=t(7293),o=t(4848);function l(e){let{className:n}=e;return(0,o.jsx)(r.A,{type:"caution",title:(0,o.jsx)(a.Rc,{}),className:(0,i.A)(n,s.G.common.unlistedBanner),children:(0,o.jsx)(a.Uh,{})})}function c(e){return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(a.AE,{}),(0,o.jsx)(l,{...e})]})}},1689:(e,n,t)=>{t.d(n,{A:()=>d});t(6540);var i=t(4164),a=t(4084),s=t(7559),r=t(7293),o=t(4848);function l(e){let{className:n}=e;return(0,o.jsx)(r.A,{type:"caution",title:(0,o.jsx)(a.Yh,{}),className:(0,i.A)(n,s.G.common.draftBanner),children:(0,o.jsx)(a.TT,{})})}var c=t(2234);function d(e){let{metadata:n}=e;const{unlisted:t,frontMatter:i}=n;return(0,o.jsxs)(o.Fragment,{children:[(t||i.unlisted)&&(0,o.jsx)(c.A,{}),i.draft&&(0,o.jsx)(l,{})]})}},3691:(e,n,t)=>{t.d(n,{A:()=>j});var i=t(6540),a=t(4164),s=t(6342);function r(e){const n=e.map((e=>({...e,parentIndex:-1,children:[]}))),t=Array(7).fill(-1);n.forEach(((e,n)=>{const i=t.slice(2,e.level);e.parentIndex=Math.max(...i),t[e.level]=n}));const i=[];return n.forEach((e=>{const{parentIndex:t,...a}=e;t>=0?n[t].children.push(a):i.push(a)})),i}function o(e){let{toc:n,minHeadingLevel:t,maxHeadingLevel:i}=e;return n.flatMap((e=>{const n=o({toc:e.children,minHeadingLevel:t,maxHeadingLevel:i});return function(e){return e.level>=t&&e.level<=i}(e)?[{...e,children:n}]:n}))}function l(e){const n=e.getBoundingClientRect();return n.top===n.bottom?l(e.parentNode):n}function c(e,n){let{anchorTopOffset:t}=n;const i=e.find((e=>l(e).top>=t));if(i){return function(e){return e.top>0&&e.bottom<window.innerHeight/2}(l(i))?i:e[e.indexOf(i)-1]??null}return e[e.length-1]??null}function d(){const e=(0,i.useRef)(0),{navbar:{hideOnScroll:n}}=(0,s.p)();return(0,i.useEffect)((()=>{e.current=n?0:document.querySelector(".navbar").clientHeight}),[n]),e}function u(e){const n=(0,i.useRef)(void 0),t=d();(0,i.useEffect)((()=>{if(!e)return()=>{};const{linkClassName:i,linkActiveClassName:a,minHeadingLevel:s,maxHeadingLevel:r}=e;function o(){const e=function(e){return Array.from(document.getElementsByClassName(e))}(i),o=function(e){let{minHeadingLevel:n,maxHeadingLevel:t}=e;const i=[];for(let a=n;a<=t;a+=1)i.push(`h${a}.anchor`);return Array.from(document.querySelectorAll(i.join()))}({minHeadingLevel:s,maxHeadingLevel:r}),l=c(o,{anchorTopOffset:t.current}),d=e.find((e=>l&&l.id===function(e){return decodeURIComponent(e.href.substring(e.href.indexOf("#")+1))}(e)));e.forEach((e=>{!function(e,t){t?(n.current&&n.current!==e&&n.current.classList.remove(a),e.classList.add(a),n.current=e):e.classList.remove(a)}(e,e===d)}))}return document.addEventListener("scroll",o),document.addEventListener("resize",o),o(),()=>{document.removeEventListener("scroll",o),document.removeEventListener("resize",o)}}),[e,t])}var m=t(8774),h=t(4848);function f(e){let{toc:n,className:t,linkClassName:i,isChild:a}=e;return n.length?(0,h.jsx)("ul",{className:a?void 0:t,children:n.map((e=>(0,h.jsxs)("li",{children:[(0,h.jsx)(m.A,{to:`#${e.id}`,className:i??void 0,dangerouslySetInnerHTML:{__html:e.value}}),(0,h.jsx)(f,{isChild:!0,toc:e.children,className:t,linkClassName:i})]},e.id)))}):null}const g=i.memo(f);function p(e){let{toc:n,className:t="table-of-contents table-of-contents__left-border",linkClassName:a="table-of-contents__link",linkActiveClassName:l,minHeadingLevel:c,maxHeadingLevel:d,...m}=e;const f=(0,s.p)(),p=c??f.tableOfContents.minHeadingLevel,x=d??f.tableOfContents.maxHeadingLevel,v=function(e){let{toc:n,minHeadingLevel:t,maxHeadingLevel:a}=e;return(0,i.useMemo)((()=>o({toc:r(n),minHeadingLevel:t,maxHeadingLevel:a})),[n,t,a])}({toc:n,minHeadingLevel:p,maxHeadingLevel:x});return u((0,i.useMemo)((()=>{if(a&&l)return{linkClassName:a,linkActiveClassName:l,minHeadingLevel:p,maxHeadingLevel:x}}),[a,l,p,x])),(0,h.jsx)(g,{toc:v,className:t,linkClassName:a,...m})}const x={tableOfContents:"tableOfContents_bqdL",docItemContainer:"docItemContainer_F8PC"},v="table-of-contents__link toc-highlight",b="table-of-contents__link--active";function j(e){let{className:n,...t}=e;return(0,h.jsx)("div",{className:(0,a.A)(x.tableOfContents,"thin-scrollbar",n),children:(0,h.jsx)(p,{...t,linkClassName:v,linkActiveClassName:b})})}},4084:(e,n,t)=>{t.d(n,{AE:()=>l,Rc:()=>r,TT:()=>d,Uh:()=>o,Yh:()=>c});t(6540);var i=t(1312),a=t(5260),s=t(4848);function r(){return(0,s.jsx)(i.A,{id:"theme.contentVisibility.unlistedBanner.title",description:"The unlisted content banner title",children:"Unlisted page"})}function o(){return(0,s.jsx)(i.A,{id:"theme.contentVisibility.unlistedBanner.message",description:"The unlisted content banner message",children:"This page is unlisted. Search engines will not index it, and only users having a direct link can access it."})}function l(){return(0,s.jsx)(a.A,{children:(0,s.jsx)("meta",{name:"robots",content:"noindex, nofollow"})})}function c(){return(0,s.jsx)(i.A,{id:"theme.contentVisibility.draftBanner.title",description:"The draft content banner title",children:"Draft page"})}function d(){return(0,s.jsx)(i.A,{id:"theme.contentVisibility.draftBanner.message",description:"The draft content banner message",children:"This page is a draft. It will only be visible in dev and be excluded from the production build."})}}}]);