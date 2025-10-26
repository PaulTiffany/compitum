 a w e s o m e   â € ”   h e r e â € ™ s   a   \ * \ * c o m p l e t e ,   s e l f - c o n t a i n e d   d e l i v e r a b l e \ * \ *   y o u   c a n   h a n d   t o   y o u r   \ * \ * G e m i n i   C L I \ * \ *   ( o r   a n y   r u n n e r   t h a t   r e a d s   a   r e p o   t a s k   f i l e )   t o   s c a f f o l d ,   i n s t a l l ,   t e s t ,   a n d   s h a d o w - r u n   \ * \ * c o m p i t u m \ * \ *   a s   a   p r o p e r   P y t h o n   p a c k a g e . 
 
 
 
 I â € ™ v e   i n c l u d e d : 
 
 
 
 \ *   A   c l e a n   \ * \ * r e p o   l a y o u t \ * \ * 
 
 \ *   A   \ * \ * G e m i n i   t a s k   f i l e \ * \ *   t o   d r i v e   s e t u p ,   t e s t s ,   a n d   a   d e m o   r o u t e 
 
 \ *   A   m i n i m a l   b u t   \ * \ * r u n n a b l e \ * \ *   c o d e b a s e   w i t h   h a r d e n e d   m a t h   ( S P D   m e t r i c ,   w h i t e n e d   K D E ,   s h a d o w   p r i c e s ) ,   a   T y p e r   \ * \ * C L I \ * \ * ,   a n d   \ * \ * p r o p e r t y   t e s t s \ * \ * 
 
 \ *   E x a m p l e   \ * \ * c o n s t r a i n t s / c o n f i g s \ * \ *   a n d   a   \ * \ * s y n t h e t i c   b e n c h m a r k \ * \ * 
 
 
 
 C o p y - p a s t e   t h e   b l o c k s   i n t o   f i l e s   a s   n a m e d ,   o r   l e t   y o u r   a g e n t   c r e a t e   t h e m   v e r b a t i m . 
 
 
 
 - - - 
 
 
 
 \ #   0 )   R e p o s i t o r y   l a y o u t 
 
 
 
 ` ` ` 
 
 c o m p i t u m / 
 
 â ” œ â ” €   p y p r o j e c t . t o m l 
 
 â ” œ â ” €   R E A D M E . m d 
 
 â ” œ â ” €   L I C E N S E 
 
 â ” œ â ” €   G e m i n i . y a m l 
 
 â ” œ â ” €   M a k e f i l e 
 
 â ” œ â ” €   c o n f i g s / 
 
 â ” ‚     â ” œ â ” €   c o n s t r a i n t s \ _ u s \ _ d e f a u l t . y a m l 
 
 â ” ‚     â ” ” â ” €   r o u t e r \ _ d e f a u l t s . y a m l 
 
 â ” œ â ” €   c o m p i t u m / 
 
 â ” ‚     â ” œ â ” €   \ _ \ _ i n i t \ _ \ _ . p y 
 
 â ” ‚     â ” œ â ” €   c l i . p y 
 
 â ” ‚     â ” œ â ” €   r o u t e r . p y 
 
 â ” ‚     â ” œ â ” €   e n e r g y . p y 
 
 â ” ‚     â ” œ â ” €   m e t r i c . p y 
 
 â ” ‚     â ” œ â ” €   c o h e r e n c e . p y 
 
 â ” ‚     â ” œ â ” €   c o n s t r a i n t s . p y 
 
 â ” ‚     â ” œ â ” €   e f f o r t \ _ q p . p y 
 
 â ” ‚     â ” œ â ” €   b o u n d a r y . p y 
 
 â ” ‚     â ” œ â ” €   c o n t r o l . p y 
 
 â ” ‚     â ” œ â ” €   p r e d i c t o r s . p y 
 
 â ” ‚     â ” œ â ” €   p g d . p y 
 
 â ” ‚     â ” œ â ” €   m o d e l s . p y 
 
 â ” ‚     â ” œ â ” €   c a p a b i l i t i e s . p y 
 
 â ” ‚     â ” ” â ” €   u t i l s . p y 
 
 â ” œ â ” €   t e s t s / 
 
 â ” ‚     â ” œ â ” €   t e s t \ _ i n v a r i a n t s . p y 
 
 â ” ‚     â ” œ â ” €   t e s t \ _ b o u n d a r y . p y 
 
 â ” ‚     â ” ” â ” €   t e s t \ _ c o n s t r a i n t s . p y 
 
 â ” ” â ” €   e x a m p l e s / 
 
 & n b s p ;     â ” œ â ” €   s y n t h \ _ b e n c h . p y 
 
 & n b s p ;     â ” ” â ” €   d e m o \ _ r o u t e . p y 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 \ #   1 )   P r o j e c t   m e t a d a t a 
 
 
 
 \ # #   ` p y p r o j e c t . t o m l ` 
 
 
 
 ` ` ` t o m l 
 
 \ [ p r o j e c t ] 
 
 n a m e   =   " c o m p i t u m " 
 
 v e r s i o n   =   " 0 . 1 . 0 " 
 
 d e s c r i p t i o n   =   " c o m p i t u m :   A   P r o d u c t i o n - R e a d y ,   G e o m e t r i c a l l y - A w a r e   A I   R o u t e r " 
 
 a u t h o r s   =   \ [ {   n a m e = " c o m p i t u m   a u t h o r s " ,   e m a i l = " d e v @ c o m p i t u m . o r g "   } ] 
 
 l i c e n s e   =   {   t e x t   =   " M I T "   } 
 
 r e a d m e   =   " R E A D M E . m d " 
 
 r e q u i r e s - p y t h o n   =   " > = 3 . 9 " 
 
 d e p e n d e n c i e s   =   \ [ 
 
 & n b s p ;   " n u m p y > = 1 . 2 4 " , 
 
 & n b s p ;   " s c i p y > = 1 . 1 0 " , 
 
 & n b s p ;   " s c i k i t - l e a r n > = 1 . 3 " , 
 
 & n b s p ;   " t y p e r > = 0 . 1 2 " , 
 
 & n b s p ;   " p y d a n t i c > = 2 . 7 " , 
 
 & n b s p ;   " p y y a m l > = 6 . 0 . 1 " 
 
 ] 
 
 
 
 \ [ p r o j e c t . o p t i o n a l - d e p e n d e n c i e s ] 
 
 d e v   =   \ [ 
 
 & n b s p ;   " p y t e s t > = 8 . 0 " , 
 
 & n b s p ;   " h y p o t h e s i s > = 6 . 9 8 " , 
 
 & n b s p ;   " p y t e s t - c o v > = 5 . 0 " , 
 
 & n b s p ;   " r u f f > = 0 . 5 . 0 " , 
 
 & n b s p ;   " m y p y > = 1 . 1 0 " , 
 
 & n b s p ;   " l i g h t g b m > = 4 . 3   ;   p y t h o n \ _ v e r s i o n   <   ' 3 . 1 3 ' " 
 
 ] 
 
 
 
 \ [ p r o j e c t . s c r i p t s ] 
 
 c o m p i t u m   =   " c o m p i t u m . c l i : a p p " 
 
 
 
 \ [ t o o l . r u f f ] 
 
 l i n e - l e n g t h   =   1 0 0 
 
 
 
 \ [ t o o l . p y t e s t . i n i \ _ o p t i o n s ] 
 
 a d d o p t s   =   " - q   - r a   - - m a x f a i l = 1 " 
 
 ` ` ` 
 
 
 
 \ # #   ` L I C E N S E ` 
 
 
 
 ` ` ` t e x t 
 
 M I T   L i c e n s e 
 
 
 
 C o p y r i g h t   ( c )   2 0 2 5   . . . 
 
 
 
 P e r m i s s i o n   i s   h e r e b y   g r a n t e d ,   f r e e   o f   c h a r g e ,   t o   a n y   p e r s o n   o b t a i n i n g   a   c o p y . . . 
 
 ` ` ` 
 
 
 
 \ # #   ` R E A D M E . m d ` 
 
 
 
 ` ` ` ` m a r k d o w n 
 
 \ #   c o m p i t u m 
 
 
 
 A   p r o d u c t i o n - r e a d y ,   g e o m e t r i c a l l y - a w a r e   A I   r o u t e r   w i t h   S P D   m e t r i c   l e a r n i n g ,   c o n s t r a i n t - a w a r e 
 
 s e l e c t i o n   ( s h a d o w   p r i c e s ) ,   m e t r i c - a w a r e   K D E   c o h e r e n c e ,   a n d   L y a p u n o v - s t a b l e   o n l i n e   u p d a t e s . 
 
 
 
 \ # #   I n s t a l l 
 
 ` ` ` b a s h 
 
 p y t h o n   - m   v e n v   . v e n v   \ & \ &   s o u r c e   . v e n v / b i n / a c t i v a t e 
 
 p i p   i n s t a l l   - e   " . \ [ d e v ] " 
 
 ` ` ` ` 
 
 
 
 \ # #   Q u i c k   d e m o 
 
 
 
 ` ` ` b a s h 
 
 c o m p i t u m   r o u t e   - - p r o m p t   " P r o v e   t h e   b i n o m i a l   i d e n t i t y   u s i n g   g e n e r a t i n g   f u n c t i o n s . " 
 
 ` ` ` 
 
 
 
 \ # #   R u n   t e s t s 
 
 
 
 ` ` ` b a s h 
 
 p y t e s t 
 
 ` ` ` 
 
 
 
 S e e   ` c o n f i g s / `   a n d   ` e x a m p l e s / `   f o r   c o n s t r a i n t s   a n d   a   s y n t h e t i c   b e n c h m a r k . 
 
 
 
 ` ` ` ` 
 
 
 
 - - - 
 
 
 
 \ #   2 )   G e m i n i   t a s k   f i l e   \ &   M a k e f i l e 
 
 
 
 \ # #   ` G e m i n i . y a m l ` 
 
 ` ` ` y a m l 
 
 v e r s i o n :   " 1 " 
 
 t a s k s : 
 
 & n b s p ;   -   i d :   s e t u p 
 
 & n b s p ;       n a m e :   C r e a t e   v e n v   a n d   i n s t a l l 
 
 & n b s p ;       s h e l l :   | 
 
 & n b s p ;           p y t h o n   - m   v e n v   . v e n v 
 
 & n b s p ;           .   . v e n v / b i n / a c t i v a t e 
 
 & n b s p ;           p i p   i n s t a l l   - U   p i p 
 
 & n b s p ;           p i p   i n s t a l l   - e   " . \ [ d e v ] " 
 
 
 
 & n b s p ;   -   i d :   l i n t 
 
 & n b s p ;       n e e d s :   \ [ s e t u p ] 
 
 & n b s p ;       s h e l l :   | 
 
 & n b s p ;           .   . v e n v / b i n / a c t i v a t e 
 
 & n b s p ;           r u f f   c h e c k   c o m p i t u m 
 
 
 
 & n b s p ;   -   i d :   t e s t 
 
 & n b s p ;       n e e d s :   \ [ s e t u p ] 
 
 & n b s p ;       s h e l l :   | 
 
 & n b s p ;           .   . v e n v / b i n / a c t i v a t e 
 
 & n b s p ;           p y t e s t 
 
 
 
 & n b s p ;   -   i d :   b e n c h - s y n t h 
 
 & n b s p ;       n e e d s :   \ [ s e t u p ] 
 
 & n b s p ;       s h e l l :   | 
 
 & n b s p ;           .   . v e n v / b i n / a c t i v a t e 
 
 & n b s p ;           p y t h o n   e x a m p l e s / s y n t h \ _ b e n c h . p y 
 
 
 
 & n b s p ;   -   i d :   r o u t e - d e m o 
 
 & n b s p ;       n e e d s :   \ [ s e t u p ] 
 
 & n b s p ;       s h e l l :   | 
 
 & n b s p ;           .   . v e n v / b i n / a c t i v a t e 
 
 & n b s p ;           c o m p i t u m   r o u t e   - - p r o m p t   " W r i t e   a   S Q L   q u e r y   t o   c o m p u t e   7 - d a y   r o l l i n g   a v e r a g e   b y   u s e r . " 
 
 ` ` ` ` 
 
 
 
 \ # #   ` M a k e f i l e ` 
 
 
 
 ` ` ` m a k e 
 
 . P H O N Y :   s e t u p   t e s t   l i n t   d e m o   b e n c h 
 
 s e t u p : 
 
 & n b s p ; 	 p y t h o n   - m   v e n v   . v e n v   \ & \ &   .   . v e n v / b i n / a c t i v a t e   \ & \ &   p i p   i n s t a l l   - U   p i p   \ & \ &   p i p   i n s t a l l   - e   " . \ [ d e v ] " 
 
 t e s t : 
 
 & n b s p ; 	 .   . v e n v / b i n / a c t i v a t e   \ & \ &   p y t e s t 
 
 l i n t : 
 
 & n b s p ; 	 .   . v e n v / b i n / a c t i v a t e   \ & \ &   r u f f   c h e c k   c o m p i t u m 
 
 d e m o : 
 
 & n b s p ; 	 .   . v e n v / b i n / a c t i v a t e   \ & \ &   c o m p i t u m   r o u t e   - - p r o m p t   " S k e t c h   a   p r o o f   f o r   A M - G M   i n e q u a l i t y . " 
 
 b e n c h : 
 
 & n b s p ; 	 .   . v e n v / b i n / a c t i v a t e   \ & \ &   p y t h o n   e x a m p l e s / s y n t h \ _ b e n c h . p y 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 \ #   3 )   C o n f i g s 
 
 
 
 \ # #   ` c o n f i g s / c o n s t r a i n t s \ _ u s \ _ d e f a u l t . y a m l ` 
 
 
 
 ` ` ` y a m l 
 
 \ #   L i n e a r   c o n s t r a i n t s   A x   < =   b   o v e r   B a n a c h   ( p r a g m a t i c )   f e a t u r e s   ( t o y   e x a m p l e ) 
 
 A : 
 
 & n b s p ;   -   \ [ 1 ,   0 ,   0 ,   0 ]       #   l a t e n c y \ _ c l a s s   < =   2 
 
 & n b s p ;   -   \ [ 0 ,   1 ,   0 ,   0 ]       #   c o s t \ _ c l a s s   < =   2 
 
 & n b s p ;   -   \ [ 0 ,   0 ,   1 ,   0 ]       #   p i i \ _ l e v e l   < =   0 
 
 & n b s p ;   -   \ [ 0 ,   0 ,   0 ,   1 ]       #   r e g i o n \ _ e u \ _ o n l y   < =   0 
 
 b :   \ [ 2 . 0 ,   2 . 0 ,   0 . 0 ,   0 . 0 ] 
 
 ` ` ` 
 
 
 
 \ # #   ` c o n f i g s / r o u t e r \ _ d e f a u l t s . y a m l ` 
 
 
 
 ` ` ` y a m l 
 
 a l p h a :   0 . 4 0 
 
 b e t a \ _ t :   0 . 2 0 
 
 b e t a \ _ c :   0 . 1 5 
 
 b e t a \ _ d :   0 . 1 5 
 
 b e t a \ _ s :   0 . 1 0 
 
 m e t r i c : 
 
 & n b s p ;   D :   3 5 
 
 & n b s p ;   r a n k :   8 
 
 & n b s p ;   d e l t a :   1 . 0 e - 3 
 
 u p d a t e \ _ s t r i d e :   8 
 
 c o l d \ _ s t a r t \ _ t h r e s h o l d :   1 6 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 \ #   4 )   C o r e   p a c k a g e   c o d e 
 
 
 
 \ # #   ` c o m p i t u m / \ _ \ _ i n i t \ _ \ _ . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 \ _ \ _ a l l \ _ \ _   =   \ [ " r o u t e r " ,   " m e t r i c " ,   " c o n s t r a i n t s " ,   " c o h e r e n c e " ,   " b o u n d a r y " ,   " c o n t r o l " ,   " e n e r g y " ] 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / c a p a b i l i t i e s . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   d a t a c l a s s e s   i m p o r t   d a t a c l a s s 
 
 f r o m   t y p i n g   i m p o r t   S e t ,   D i c t ,   A n y 
 
 
 
 @ d a t a c l a s s 
 
 c l a s s   C a p a b i l i t i e s : 
 
 & n b s p ;       r e g i o n s :   S e t \ [ s t r ] 
 
 & n b s p ;       t o o l s \ _ a l l o w e d :   S e t \ [ s t r ] 
 
 & n b s p ;       d e t e r m i n i s t i c :   b o o l   =   F a l s e 
 
 
 
 & n b s p ;       d e f   s u p p o r t s ( s e l f ,   p g d \ _ v e c t o r :   A n y ,   c o n t e x t :   D i c t \ [ s t r ,   A n y ]   |   N o n e   =   N o n e )   - >   b o o l : 
 
 & n b s p ;               #   H o o k   f o r   m o d e l - s p e c i f i c   g a t e s ;   e x t e n d   a s   n e e d e d . 
 
 & n b s p ;               #   E x a m p l e :   b l o c k   i f   c o n t e x t \ [ " r e g i o n " ]   n o t   i n   s e l f . r e g i o n s 
 
 & n b s p ;               r e t u r n   T r u e 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / m o d e l s . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   d a t a c l a s s e s   i m p o r t   d a t a c l a s s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   . c a p a b i l i t i e s   i m p o r t   C a p a b i l i t i e s 
 
 
 
 @ d a t a c l a s s 
 
 c l a s s   M o d e l : 
 
 & n b s p ;       n a m e :   s t r 
 
 & n b s p ;       c e n t e r :   n p . n d a r r a y     #   c e n t e r   i n   R i e m a n n i a n   f e a t u r e   s p a c e 
 
 & n b s p ;       c a p a b i l i t i e s :   C a p a b i l i t i e s 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / u t i l s . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   h a s h l i b 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   t y p i n g   i m p o r t   D i c t ,   T u p l e 
 
 
 
 d e f   s p l i t \ _ f e a t u r e s ( x :   D i c t \ [ s t r ,   f l o a t ] )   - >   T u p l e \ [ n p . n d a r r a y ,   n p . n d a r r a y ] : 
 
 & n b s p ;       #   R i e m a n n i a n :   e v e r y t h i n g   e x c e p t   p r a g \ _ \ * ,   B a n a c h :   p r a g \ _ \ *   o n l y 
 
 & n b s p ;       x R   =   \ [ v   f o r   k ,   v   i n   x . i t e m s ( )   i f   n o t   k . s t a r t s w i t h ( " p r a g \ _ " ) ] 
 
 & n b s p ;       x B   =   \ [ v   f o r   k ,   v   i n   x . i t e m s ( )   i f   k . s t a r t s w i t h ( " p r a g \ _ " ) ] 
 
 & n b s p ;       r e t u r n   n p . a r r a y ( x R ,   f l o a t ) ,   n p . a r r a y ( x B ,   f l o a t ) 
 
 
 
 d e f   p g d \ _ h a s h ( p r o m p t :   s t r )   - >   s t r : 
 
 & n b s p ;       r e t u r n   h a s h l i b . m d 5 ( p r o m p t . e n c o d e ( ) ) . h e x d i g e s t ( ) 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / m e t r i c . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   t y p i n g   i m p o r t   O p t i o n a l ,   T u p l e 
 
 f r o m   s c i p y . l i n a l g   i m p o r t   c h o l e s k y ,   L i n A l g E r r o r 
 
 f r o m   s k l e a r n . c o v a r i a n c e   i m p o r t   L e d o i t W o l f 
 
 
 
 c l a s s   S y m b o l i c M a n i f o l d M e t r i c : 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ,   D :   i n t ,   r a n k :   i n t ,   d e l t a :   f l o a t   =   1 e - 3 ) : 
 
 & n b s p ;               s e l f . D ,   s e l f . r a n k ,   s e l f . d e l t a   =   D ,   r a n k ,   d e l t a 
 
 & n b s p ;               s e l f . L   =   n p . r a n d o m . r a n d n ( D ,   r a n k )   \ *   0 . 0 1 
 
 & n b s p ;               s e l f . W :   O p t i o n a l \ [ n p . n d a r r a y ]   =   N o n e 
 
 & n b s p ;               s e l f . s h r i n k   =   L e d o i t W o l f ( ) 
 
 & n b s p ;               s e l f . w h i t e n e d \ _ r e s i d u a l s :   l i s t \ [ n p . n d a r r a y ]   =   \ [ ] 
 
 
 
 & n b s p ;       d e f   m e t r i c \ _ m a t r i x ( s e l f )   - >   n p . n d a r r a y : 
 
 & n b s p ;               r e t u r n   s e l f . L   @   s e l f . L . T   +   s e l f . d e l t a   \ *   n p . e y e ( s e l f . D ) 
 
 
 
 & n b s p ;       d e f   \ _ u p d a t e \ _ c h o l e s k y ( s e l f )   - >   n p . n d a r r a y : 
 
 & n b s p ;               t r y : 
 
 & n b s p ;                       s e l f . W   =   c h o l e s k y ( s e l f . m e t r i c \ _ m a t r i x ( ) ,   l o w e r = F a l s e ) 
 
 & n b s p ;               e x c e p t   L i n A l g E r r o r : 
 
 & n b s p ;                       s e l f . d e l t a   =   m i n ( s e l f . d e l t a   \ *   2 . 0 ,   1 e - 1 ) 
 
 & n b s p ;                       s e l f . W   =   c h o l e s k y ( s e l f . m e t r i c \ _ m a t r i x ( ) ,   l o w e r = F a l s e ) 
 
 & n b s p ;               r e t u r n   s e l f . W 
 
 
 
 & n b s p ;       d e f   d i s t a n c e ( s e l f ,   x :   n p . n d a r r a y ,   m u :   n p . n d a r r a y )   - >   T u p l e \ [ f l o a t ,   f l o a t ] : 
 
 & n b s p ;               i f   s e l f . W   i s   N o n e : 
 
 & n b s p ;                       s e l f . \ _ u p d a t e \ _ c h o l e s k y ( ) 
 
 & n b s p ;               z   =   x   -   m u 
 
 & n b s p ;               w z   =   s e l f . W   @   z 
 
 & n b s p ;               d   =   f l o a t ( n p . l i n a l g . n o r m ( w z ) ) 
 
 & n b s p ;               i f   l e n ( s e l f . w h i t e n e d \ _ r e s i d u a l s )   >   s e l f . r a n k : 
 
 & n b s p ;                       c o v   =   s e l f . s h r i n k . f i t ( n p . a r r a y ( s e l f . w h i t e n e d \ _ r e s i d u a l s ) ) . c o v a r i a n c e \ _ 
 
 & n b s p ;                       s i g m a   =   f l o a t ( n p . s q r t ( m a x ( w z . T   @   c o v   @   w z ,   0 . 0 ) ) ) 
 
 & n b s p ;               e l s e : 
 
 & n b s p ;                       s i g m a   =   0 . 1 
 
 & n b s p ;               r e t u r n   d ,   s i g m a 
 
 
 
 & n b s p ;       d e f   u p d a t e \ _ s p d ( s e l f ,   x :   n p . n d a r r a y ,   m u :   n p . n d a r r a y ,   b e t a \ _ d :   f l o a t ,   d :   f l o a t ,   e t a :   f l o a t , 
 
 & n b s p ;                                     s r m f \ _ c o n t r o l l e r )   - >   f l o a t : 
 
 & n b s p ;               z   =   x   -   m u 
 
 & n b s p ;               A   =   - ( b e t a \ _ d   /   ( 2   \ *   m a x ( d ,   1 e - 8 ) ) )   \ *   n p . o u t e r ( z ,   z )     #   d U / d M 
 
 & n b s p ;               g r a d \ _ L   =   2   \ *   A   @   s e l f . L 
 
 & n b s p ;               g r a d \ _ n o r m   =   f l o a t ( n p . l i n a l g . n o r m ( g r a d \ _ L ,   2 ) ) 
 
 & n b s p ;               e t a \ _ c a p ,   \ _   =   s r m f \ _ c o n t r o l l e r . u p d a t e ( d \ _ s t a r = d ,   g r a d \ _ n o r m = g r a d \ _ n o r m ) 
 
 & n b s p ;               s e l f . L   - =   m i n ( e t a ,   e t a \ _ c a p )   \ *   g r a d \ _ L 
 
 & n b s p ;               f n o r m   =   n p . l i n a l g . n o r m ( s e l f . L ,   " f r o " ) 
 
 & n b s p ;               i f   f n o r m   >   1 0 . 0 : 
 
 & n b s p ;                       s e l f . L   \ * =   ( 1 0 . 0   /   f n o r m ) 
 
 & n b s p ;               W   =   s e l f . \ _ u p d a t e \ _ c h o l e s k y ( ) 
 
 & n b s p ;               s e l f . w h i t e n e d \ _ r e s i d u a l s . a p p e n d ( W   @   z ) 
 
 & n b s p ;               i f   l e n ( s e l f . w h i t e n e d \ _ r e s i d u a l s )   >   1 0 0 : 
 
 & n b s p ;                       s e l f . w h i t e n e d \ _ r e s i d u a l s . p o p ( 0 ) 
 
 & n b s p ;               r e t u r n   g r a d \ _ n o r m 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / c o h e r e n c e . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   c o l l e c t i o n s   i m p o r t   d e f a u l t d i c t 
 
 f r o m   s k l e a r n . n e i g h b o r s   i m p o r t   K e r n e l D e n s i t y 
 
 
 
 c l a s s   W e i g h t e d R e s e r v o i r : 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ,   k = 1 0 0 0 ,   r n g = N o n e ) : 
 
 & n b s p ;               s e l f . k ,   s e l f . b u f ,   s e l f . t o t \ _ w   =   k ,   \ [ ] ,   0 . 0 
 
 & n b s p ;               s e l f . r n g   =   r n g   o r   n p . r a n d o m . d e f a u l t \ _ r n g ( ) 
 
 
 
 & n b s p ;       d e f   a d d ( s e l f ,   x :   n p . n d a r r a y ,   w :   f l o a t ) : 
 
 & n b s p ;               w   =   m a x ( f l o a t ( w ) ,   1 e - 6 ) 
 
 & n b s p ;               s e l f . t o t \ _ w   + =   w 
 
 & n b s p ;               i f   l e n ( s e l f . b u f )   <   s e l f . k : 
 
 & n b s p ;                       s e l f . b u f . a p p e n d ( ( x . c o p y ( ) ,   w ) ) 
 
 & n b s p ;               e l s e : 
 
 & n b s p ;                       j   =   i n t ( s e l f . r n g . i n t e g e r s ( 0 ,   i n t ( s e l f . t o t \ _ w ) ) ) 
 
 & n b s p ;                       i f   j   <   s e l f . k : 
 
 & n b s p ;                               s e l f . b u f \ [ j ]   =   ( x . c o p y ( ) ,   w ) 
 
 
 
 c l a s s   C o h e r e n c e F u n c t i o n a l : 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ,   k = 1 0 0 0 ) : 
 
 & n b s p ;               s e l f . r e s   =   d e f a u l t d i c t ( l a m b d a :   W e i g h t e d R e s e r v o i r ( k ) ) 
 
 & n b s p ;               s e l f . k d e \ _ c a c h e :   d i c t \ [ s t r ,   K e r n e l D e n s i t y ]   =   { } 
 
 
 
 & n b s p ;       d e f   u p d a t e ( s e l f ,   m o d e l \ _ n a m e :   s t r ,   x w :   n p . n d a r r a y ,   s u c c e s s :   f l o a t ) : 
 
 & n b s p ;               s e l f . r e s \ [ m o d e l \ _ n a m e ] . a d d ( x w ,   s u c c e s s ) 
 
 & n b s p ;               s e l f . k d e \ _ c a c h e . p o p ( m o d e l \ _ n a m e ,   N o n e ) 
 
 
 
 & n b s p ;       d e f   \ _ f i t ( s e l f ,   m o d e l \ _ n a m e :   s t r )   - >   K e r n e l D e n s i t y   |   N o n e : 
 
 & n b s p ;               b u f   =   s e l f . r e s \ [ m o d e l \ _ n a m e ] . b u f 
 
 & n b s p ;               i f   l e n ( b u f )   <   1 0 : 
 
 & n b s p ;                       r e t u r n   N o n e 
 
 & n b s p ;               X   =   n p . s t a c k ( \ [ x   f o r   x ,   \ _   i n   b u f ] ,   a x i s = 0 ) 
 
 & n b s p ;               w   =   n p . a r r a y ( \ [ w t   f o r   \ _ ,   w t   i n   b u f ] ,   f l o a t ) 
 
 & n b s p ;               #   S c o t t   r u l e   o n   w h i t e n e d   c o o r d s 
 
 & n b s p ;               n ,   d   =   X . s h a p e 
 
 & n b s p ;               b w   =   n   \ * \ *   ( - 1 . 0   /   ( d   +   4 ) ) 
 
 & n b s p ;               k d e   =   K e r n e l D e n s i t y ( k e r n e l = " g a u s s i a n " ,   b a n d w i d t h = b w ) . f i t ( X ,   s a m p l e \ _ w e i g h t = w   /   w . s u m ( ) ) 
 
 & n b s p ;               s e l f . k d e \ _ c a c h e \ [ m o d e l \ _ n a m e ]   =   k d e 
 
 & n b s p ;               r e t u r n   k d e 
 
 
 
 & n b s p ;       d e f   l o g \ _ e v i d e n c e ( s e l f ,   m o d e l \ _ n a m e :   s t r ,   x w :   n p . n d a r r a y )   - >   f l o a t : 
 
 & n b s p ;               k d e   =   s e l f . k d e \ _ c a c h e . g e t ( m o d e l \ _ n a m e )   o r   s e l f . \ _ f i t ( m o d e l \ _ n a m e ) 
 
 & n b s p ;               i f   k d e   i s   N o n e : 
 
 & n b s p ;                       r e t u r n   0 . 0 
 
 & n b s p ;               v a l   =   f l o a t ( k d e . s c o r e \ _ s a m p l e s ( \ [ x w ] ) \ [ 0 ] ) 
 
 & n b s p ;               r e t u r n   f l o a t ( n p . c l i p ( v a l ,   - 1 0 . 0 ,   1 0 . 0 ) ) 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / c o n s t r a i n t s . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   t y p i n g   i m p o r t   A n y ,   D i c t ,   L i s t ,   T u p l e 
 
 
 
 c l a s s   R e f l e c t i v e C o n s t r a i n t S o l v e r : 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ,   A :   n p . n d a r r a y ,   b :   n p . n d a r r a y ) : 
 
 & n b s p ;               s e l f . A ,   s e l f . b   =   A ,   b 
 
 & n b s p ;               s e l f . l a s t \ _ v i a b l e \ _ m o d e l s :   L i s t \ [ A n y ]   =   \ [ ] 
 
 
 
 & n b s p ;       d e f   \ _ i s \ _ f e a s i b l e ( s e l f ,   m o d e l :   A n y ,   p g d \ _ b a n a c h :   n p . n d a r r a y )   - >   b o o l : 
 
 & n b s p ;               i f   n o t   n p . a l l ( s e l f . A   @   p g d \ _ b a n a c h   < =   s e l f . b   +   1 e - 1 0 ) : 
 
 & n b s p ;                       r e t u r n   F a l s e 
 
 & n b s p ;               r e t u r n   m o d e l . c a p a b i l i t i e s . s u p p o r t s ( p g d \ _ b a n a c h ) 
 
 
 
 & n b s p ;       d e f   s e l e c t ( s e l f ,   p g d \ _ b a n a c h :   n p . n d a r r a y ,   m o d e l s :   L i s t \ [ A n y ] , 
 
 & n b s p ;                             u t i l i t i e s :   D i c t \ [ A n y ,   f l o a t ] ,   e p s :   f l o a t   =   1 e - 3 )   - >   T u p l e \ [ A n y ,   D i c t ] : 
 
 & n b s p ;               v i a b l e   =   \ [ m   f o r   m   i n   m o d e l s   i f   s e l f . \ _ i s \ _ f e a s i b l e ( m ,   p g d \ _ b a n a c h ) ] 
 
 & n b s p ;               s e l f . l a s t \ _ v i a b l e \ _ m o d e l s   =   v i a b l e 
 
 & n b s p ;               i f   n o t   v i a b l e : 
 
 & n b s p ;                       m \ _ s t a r   =   m a x ( m o d e l s ,   k e y = l a m b d a   m :   u t i l i t i e s \ [ m ] ) 
 
 & n b s p ;                       r e t u r n   m \ _ s t a r ,   { " f e a s i b l e " :   F a l s e ,   " m i n i m a l \ _ v i o l a t i o n " :   T r u e , 
 
 & n b s p ;                                                       " b i n d i n g \ _ c o n s t r a i n t s " :   \ [ ] ,   " s h a d o w \ _ p r i c e s " :   { } } 
 
 
 
 & n b s p ;               m \ _ s t a r   =   m a x ( v i a b l e ,   k e y = l a m b d a   m :   u t i l i t i e s \ [ m ] ) 
 
 
 
 & n b s p ;               l a m b d a s   =   { } 
 
 & n b s p ;               f o r   j   i n   r a n g e ( s e l f . b . s i z e ) : 
 
 & n b s p ;                       b \ _ r e l a x e d   =   s e l f . b . c o p y ( ) ;   b \ _ r e l a x e d \ [ j ]   + =   e p s 
 
 & n b s p ;                       #   i f   r e l a x a t i o n   c h a n g e s   f e a s i b i l i t y   o f   b e t t e r   c o m p e t i t o r s ,   e s t i m a t e   â ˆ ‚ U / â ˆ ‚ b \ _ j 
 
 & n b s p ;                       b e s t \ _ u t i l   =   u t i l i t i e s \ [ m \ _ s t a r ] 
 
 & n b s p ;                       f o r   c o m p   i n   m o d e l s : 
 
 & n b s p ;                               i f   c o m p   i n   v i a b l e   o r   u t i l i t i e s \ [ c o m p ]   < =   b e s t \ _ u t i l : 
 
 & n b s p ;                                       c o n t i n u e 
 
 & n b s p ;                               o k   =   n p . a l l ( s e l f . A   @   p g d \ _ b a n a c h   < =   b \ _ r e l a x e d   +   1 e - 1 0 )   a n d   c o m p . c a p a b i l i t i e s . s u p p o r t s ( p g d \ _ b a n a c h ) 
 
 & n b s p ;                               i f   o k : 
 
 & n b s p ;                                       b e s t \ _ u t i l   =   m a x ( b e s t \ _ u t i l ,   u t i l i t i e s \ [ c o m p ] ) 
 
 & n b s p ;                       l a m b d a s \ [ f " l a m b d a \ _ { j } " ]   =   m a x ( 0 . 0 ,   ( b e s t \ _ u t i l   -   u t i l i t i e s \ [ m \ _ s t a r ] )   /   e p s ) 
 
 
 
 & n b s p ;               b i n d i n g   =   \ [ j   f o r   j ,   v a l   i n   e n u m e r a t e ( s e l f . A   @   p g d \ _ b a n a c h )   i f   v a l   > =   s e l f . b \ [ j ]   -   1 e - 9 ] 
 
 & n b s p ;               r e t u r n   m \ _ s t a r ,   { " f e a s i b l e " :   T r u e ,   " m i n i m a l \ _ v i o l a t i o n " :   F a l s e , 
 
 & n b s p ;                                               " b i n d i n g \ _ c o n s t r a i n t s " :   b i n d i n g ,   " s h a d o w \ _ p r i c e s " :   l a m b d a s } 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / e f f o r t \ _ q p . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 d e f   s o l v e \ _ e f f o r t \ _ 1 d ( q 0 ,   q 1 ,   t 0 ,   t 1 ,   c 0 ,   c 1 ,   b e t a ) : 
 
 & n b s p ;       " " " 
 
 & n b s p ;       L i n e a r i z e d   e f f o r t   e â ˆ ˆ \ [ 0 , 1 ]   a r o u n d   e 0 .   R e t u r n s   e \ _ s t a r   a n d   b o x   m u l t i p l i e r s . 
 
 & n b s p ;       U ( e )   =   Î ± ( q 0 + q 1   e )   -   Î ² t ( t 0 + t 1   e )   -   Î ² c ( c 0 + c 1   e )   +   c o n s t 
 
 & n b s p ;       " " " 
 
 & n b s p ;       a l p h a ,   b t ,   b c   =   b e t a 
 
 & n b s p ;       g r a d   =   a l p h a \ * q 1   -   b t \ * t 1   -   b c \ * c 1 
 
 & n b s p ;       e \ _ s t a r   =   1 . 0   i f   g r a d   >   0   e l s e   0 . 0 
 
 & n b s p ;       l a m \ _ l o w     =   m a x ( 0 . 0 ,   - g r a d )   i f   e \ _ s t a r   = =   0 . 0   e l s e   0 . 0 
 
 & n b s p ;       l a m \ _ h i g h   =   m a x ( 0 . 0 ,     g r a d )   i f   e \ _ s t a r   = =   1 . 0   e l s e   0 . 0 
 
 & n b s p ;       r e t u r n   f l o a t ( e \ _ s t a r ) ,   { " l a m b d a \ _ l o w " :   l a m \ _ l o w ,   " l a m b d a \ _ h i g h " :   l a m \ _ h i g h } 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / b o u n d a r y . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   t y p i n g   i m p o r t   D i c t ,   A n y 
 
 
 
 c l a s s   B o u n d a r y A n a l y z e r : 
 
 & n b s p ;       d e f   a n a l y z e ( s e l f ,   u t i l i t i e s :   D i c t \ [ s t r ,   f l o a t ] ,   u \ _ s i g m a :   D i c t \ [ s t r ,   f l o a t ] )   - >   D i c t \ [ s t r ,   A n y ] : 
 
 & n b s p ;               i f   l e n ( u t i l i t i e s )   <   2 : 
 
 & n b s p ;                       r e t u r n   { " i s \ _ b o u n d a r y " :   F a l s e ,   " r e a s o n " :   " i n s u f f i c i e n t \ _ m o d e l s " } 
 
 & n b s p ;               i t e m s   =   s o r t e d ( u t i l i t i e s . i t e m s ( ) ,   k e y = l a m b d a   k v :   k v \ [ 1 ] ,   r e v e r s e = T r u e ) 
 
 & n b s p ;               ( m 1 ,   u 1 ) ,   ( m 2 ,   u 2 )   =   i t e m s \ [ 0 ] ,   i t e m s \ [ 1 ] 
 
 & n b s p ;               g a p   =   u 1   -   u 2 
 
 & n b s p ;               a r r   =   n p . a r r a y ( \ [ u   f o r   \ _ ,   u   i n   i t e m s ] ) 
 
 & n b s p ;               p r o b s   =   n p . e x p ( a r r   -   u 1 ) ;   p r o b s   / =   p r o b s . s u m ( ) 
 
 & n b s p ;               e n t r o p y   =   - f l o a t ( n p . s u m ( p r o b s   \ *   n p . l o g ( p r o b s   +   1 e - 1 2 ) ) ) 
 
 & n b s p ;               s i g m a   =   f l o a t ( u \ _ s i g m a . g e t ( m 1 ,   0 . 0 ) ) 
 
 & n b s p ;               i s \ _ b o u n d a r y   =   ( g a p   <   0 . 0 5   o r   e n t r o p y   >   0 . 6 5 )   a n d   ( s i g m a   >   0 . 1 2 ) 
 
 & n b s p ;               r e t u r n   { " w i n n e r " :   m 1 ,   " r u n n e r \ _ u p " :   m 2 ,   " u t i l i t y \ _ g a p " :   f l o a t ( g a p ) , 
 
 & n b s p ;                               " e n t r o p y " :   f l o a t ( e n t r o p y ) ,   " u n c e r t a i n t y " :   s i g m a ,   " i s \ _ b o u n d a r y " :   b o o l ( i s \ _ b o u n d a r y ) } 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / c o n t r o l . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   t y p i n g   i m p o r t   T u p l e ,   D i c t 
 
 
 
 c l a s s   S R M F C o n t r o l l e r : 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ,   k a p p a :   f l o a t   =   0 . 1 ,   r 0 :   f l o a t   =   1 . 0 ) : 
 
 & n b s p ;               s e l f . k a p p a   =   k a p p a 
 
 & n b s p ;               s e l f . r   =   r 0 
 
 & n b s p ;               s e l f . e m a \ _ d   =   0 . 0 
 
 
 
 & n b s p ;       d e f   u p d a t e ( s e l f ,   d \ _ s t a r :   f l o a t ,   g r a d \ _ n o r m :   f l o a t )   - >   T u p l e \ [ f l o a t ,   D i c t \ [ s t r ,   f l o a t ] ] : 
 
 & n b s p ;               s e l f . e m a \ _ d   =   0 . 9 \ * s e l f . e m a \ _ d   +   0 . 1 \ * f l o a t ( d \ _ s t a r ) 
 
 & n b s p ;               e t a \ _ c a p   =   s e l f . k a p p a   /   ( f l o a t ( g r a d \ _ n o r m )   +   1 e - 6 ) 
 
 & n b s p ;               i f   s e l f . e m a \ _ d   >   1 . 5 \ * s e l f . r : 
 
 & n b s p ;                       s e l f . r   \ * =   0 . 8 
 
 & n b s p ;               e l i f   s e l f . e m a \ _ d   <   0 . 7 \ * s e l f . r : 
 
 & n b s p ;                       s e l f . r   \ * =   1 . 1 
 
 & n b s p ;               s e l f . r   =   f l o a t ( n p . c l i p ( s e l f . r ,   0 . 2 ,   5 . 0 ) ) 
 
 & n b s p ;               r e t u r n   f l o a t ( e t a \ _ c a p ) ,   { " t r u s t \ _ r a d i u s " :   s e l f . r ,   " d r i f t \ _ e m a " :   s e l f . e m a \ _ d } 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / p r e d i c t o r s . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   s k l e a r n . i s o t o n i c   i m p o r t   I s o t o n i c R e g r e s s i o n 
 
 f r o m   s k l e a r n . e n s e m b l e   i m p o r t   G r a d i e n t B o o s t i n g R e g r e s s o r 
 
 
 
 c l a s s   C a l i b r a t e d P r e d i c t o r : 
 
 & n b s p ;       " " " 
 
 & n b s p ;       C a l i b r a t e d   r e g r e s s o r   w i t h   q u a n t i l e   b o u n d s   ( p 5 , p 9 5 ) . 
 
 & n b s p ;       F o r   l a t e n c y / c o s t :   c o n s i d e r   e n a b l i n g   m o n o t o n i c   c o n s t r a i n t s   v i a   L i g h t G B M   w h e n   a v a i l a b l e . 
 
 & n b s p ;       " " " 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ) : 
 
 & n b s p ;               s e l f . b a s e   =   G r a d i e n t B o o s t i n g R e g r e s s o r ( r a n d o m \ _ s t a t e = 4 2 ) 
 
 & n b s p ;               s e l f . i s o   =   I s o t o n i c R e g r e s s i o n ( o u t \ _ o f \ _ b o u n d s = " c l i p " ) 
 
 & n b s p ;               s e l f . q 0 5   =   G r a d i e n t B o o s t i n g R e g r e s s o r ( l o s s = " q u a n t i l e " ,   a l p h a = 0 . 0 5 ,   r a n d o m \ _ s t a t e = 4 1 ) 
 
 & n b s p ;               s e l f . q 9 5   =   G r a d i e n t B o o s t i n g R e g r e s s o r ( l o s s = " q u a n t i l e " ,   a l p h a = 0 . 9 5 ,   r a n d o m \ _ s t a t e = 4 3 ) 
 
 & n b s p ;               s e l f . f i t t e d   =   F a l s e 
 
 
 
 & n b s p ;       d e f   f i t ( s e l f ,   X :   n p . n d a r r a y ,   y :   n p . n d a r r a y ) : 
 
 & n b s p ;               s e l f . b a s e . f i t ( X ,   y ) 
 
 & n b s p ;               r a w   =   s e l f . b a s e . p r e d i c t ( X ) 
 
 & n b s p ;               s e l f . i s o . f i t ( r a w ,   y ) 
 
 & n b s p ;               s e l f . q 0 5 . f i t ( X ,   y ) 
 
 & n b s p ;               s e l f . q 9 5 . f i t ( X ,   y ) 
 
 & n b s p ;               s e l f . f i t t e d   =   T r u e 
 
 
 
 & n b s p ;       d e f   p r e d i c t ( s e l f ,   X :   n p . n d a r r a y ) : 
 
 & n b s p ;               r a w   =   s e l f . b a s e . p r e d i c t ( X ) 
 
 & n b s p ;               y   =   s e l f . i s o . t r a n s f o r m ( r a w ) 
 
 & n b s p ;               l o   =   s e l f . q 0 5 . p r e d i c t ( X ) 
 
 & n b s p ;               h i   =   s e l f . q 9 5 . p r e d i c t ( X ) 
 
 & n b s p ;               r e t u r n   y ,   l o ,   h i 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / p g d . p y ` 
 
 
 
 ` ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   r e 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   t y p i n g   i m p o r t   D i c t 
 
 
 
 c l a s s   R e g e x P r o m p t E x t r a c t o r : 
 
 & n b s p ;       " " " 
 
 & n b s p ;       F a s t ,   r e g e x - f i r s t   e x t r a c t o r   ( s p a C y   o p t i o n a l ) .   R e t u r n s   a   s t a b l e   3 5 D   R i e m a n n i a n   v e c t o r 
 
 & n b s p ;       p l u s   a   s m a l l   B a n a c h   v e c t o r   a t t a c h e d   s e p a r a t e l y   b y   t h e   c a l l e r   i f   d e s i r e d . 
 
 & n b s p ;       " " " 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ) : 
 
 & n b s p ;               s e l f . \ _ r \ _ k e y s   =   \ [ f " s y n \ _ { i } "   f o r   i   i n   r a n g e ( 6 ) ]   +   \ \ 
 
 & n b s p ;                                             \ [ f " m a t h \ _ { i } "   f o r   i   i n   r a n g e ( 8 ) ]   +   \ \ 
 
 & n b s p ;                                             \ [ f " c o d e \ _ { i } "   f o r   i   i n   r a n g e ( 7 ) ]   +   \ \ 
 
 & n b s p ;                                             \ [ f " s e m \ _ { i } "   f o r   i   i n   r a n g e ( 6 ) ]   +   \ \ 
 
 & n b s p ;                                             \ [ f " a u x \ _ { i } "   f o r   i   i n   r a n g e ( 8 ) ]     #   p a d   t o   3 5   i f   s o m e   g r o u p s   a r e   l i g h t 
 
 
 
 & n b s p ;       d e f   e x t r a c t \ _ f e a t u r e s ( s e l f ,   p r o m p t :   s t r )   - >   D i c t \ [ s t r ,   f l o a t ] : 
 
 & n b s p ;               f e a t s :   D i c t \ [ s t r ,   f l o a t ]   =   { } 
 
 & n b s p ;               #   s y n t a c t i c   ( c h e a p   p r o x i e s ) 
 
 & n b s p ;               s e n t s   =   \ [ s   f o r   s   i n   r e . s p l i t ( r " \ [ . ! ? ] \ \ s + " ,   p r o m p t )   i f   s ] 
 
 & n b s p ;               f e a t s \ [ " s y n \ _ 0 " ]   =   f l o a t ( n p . m e a n ( \ [ l e n ( s . s p l i t ( ) )   f o r   s   i n   s e n t s ] ) )   i f   s e n t s   e l s e   0 . 0 
 
 & n b s p ;               f e a t s \ [ " s y n \ _ 1 " ]   =   f l o a t ( n p . s t d ( \ [ l e n ( s . s p l i t ( ) )   f o r   s   i n   s e n t s ] ) )   i f   s e n t s   e l s e   0 . 0 
 
 & n b s p ;               f e a t s \ [ " s y n \ _ 2 " ]   =   f l o a t ( l e n ( s e n t s ) ) 
 
 & n b s p ;               f e a t s \ [ " s y n \ _ 3 " ]   =   f l o a t ( p r o m p t . c o u n t ( " , " ) ) 
 
 & n b s p ;               f e a t s \ [ " s y n \ _ 4 " ]   =   f l o a t ( p r o m p t . c o u n t ( " ; " ) ) 
 
 & n b s p ;               f e a t s \ [ " s y n \ _ 5 " ]   =   f l o a t ( m i n ( l e n ( p r o m p t ) ,   4 0 9 6 ) )     #   p r o x y   f o r   l e n g t h 
 
 
 
 & n b s p ;               #   m a t h 
 
 & n b s p ;               m a t h \ _ o p s   =   l e n ( r e . f i n d a l l ( r " \ [ â ˆ ‘ â ˆ  â ˆ « â ˆ ‚ â ˆ ‡ â ‰ ¤ â ‰ ¥ â ‰   â ‰ ˆ ] | \ \ \ \ ( s u m | i n t | p r o d | f r a c | c d o t ) " ,   p r o m p t ) ) 
 
 & n b s p ;               l a t e x   =   l e n ( r e . f i n d a l l ( r " \ \ $ \ [ ^ $ ] + \ \ $ | \ \ \ \ b e g i n \ \ { e q u a t i o n \ \ } " ,   p r o m p t ) ) 
 
 & n b s p ;               f e a t s   | =   { 
 
 & n b s p ;                       " m a t h \ _ 0 " :   m a t h \ _ o p s , 
 
 & n b s p ;                       " m a t h \ _ 1 " :   l a t e x , 
 
 & n b s p ;                       " m a t h \ _ 2 " :   f l o a t ( l e n ( r e . f i n d a l l ( r " \ \ b p r o v e | d e r i v e | c o m p u t e | s o l v e \ \ b " ,   p r o m p t ,   r e . I ) ) ) , 
 
 & n b s p ;                       " m a t h \ _ 3 " :   f l o a t ( l e n ( r e . f i n d a l l ( r " \ [ 0 - 9 ] + ( \ \ . \ [ 0 - 9 ] + ) ? " ,   p r o m p t ) ) ) , 
 
 & n b s p ;                       " m a t h \ _ 4 " :   f l o a t ( p r o m p t . c o u n t ( " ^ " ) + p r o m p t . c o u n t ( " \ _ " ) ) , 
 
 & n b s p ;                       " m a t h \ _ 5 " :   f l o a t ( " t h e o r e m "   i n   p r o m p t . l o w e r ( ) ) , 
 
 & n b s p ;                       " m a t h \ _ 6 " :   f l o a t ( " l e m m a "   i n   p r o m p t . l o w e r ( ) ) , 
 
 & n b s p ;                       " m a t h \ _ 7 " :   f l o a t ( " p r o o f "   i n   p r o m p t . l o w e r ( ) ) , 
 
 & n b s p ;               } 
 
 
 
 & n b s p ;               #   c o d e 
 
 & n b s p ;               c o d e \ _ b l o c k s   =   l e n ( r e . f i n d a l l ( r " ` ` ` \ [ \ \ s \ \ S ] \ * ? ` ` ` " ,   p r o m p t ) ) 
 
 & n b s p ;               l a n g \ _ h i t s   =   l e n ( r e . f i n d a l l ( r " \ \ b ( p y t h o n | s q l | j a v a s c r i p t | c p p | j a v a | r u s t | g o ) \ \ b " ,   p r o m p t ,   r e . I ) ) 
 
 & n b s p ;               f e a t s   | =   { 
 
 & n b s p ;                       " c o d e \ _ 0 " :   f l o a t ( c o d e \ _ b l o c k s ) , 
 
 & n b s p ;                       " c o d e \ _ 1 " :   f l o a t ( l a n g \ _ h i t s ) , 
 
 & n b s p ;                       " c o d e \ _ 2 " :   f l o a t ( l e n ( r e . f i n d a l l ( r " \ \ b f o r | w h i l e | i f | e l s e | t r y | c a t c h | e x c e p t \ \ b " ,   p r o m p t ,   r e . I ) ) ) , 
 
 & n b s p ;                       " c o d e \ _ 3 " :   f l o a t ( l e n ( r e . f i n d a l l ( r " \ [ { } ( ) ; ] " ,   p r o m p t ) ) ) , 
 
 & n b s p ;                       " c o d e \ _ 4 " :   f l o a t ( " c l a s s   "   i n   p r o m p t   o r   " d e f   "   i n   p r o m p t ) , 
 
 & n b s p ;                       " c o d e \ _ 5 " :   f l o a t ( " S E L E C T   "   i n   p r o m p t . u p p e r ( ) ) , 
 
 & n b s p ;                       " c o d e \ _ 6 " :   f l o a t ( " i m p o r t   "   i n   p r o m p t ) , 
 
 & n b s p ;               } 
 
 
 
 & n b s p ;               #   s e m a n t i c   p r o x i e s 
 
 & n b s p ;               t o k e n s   =   p r o m p t . s p l i t ( ) 
 
 & n b s p ;               d i f f s   =   \ [ a b s ( l e n ( t o k e n s \ [ i + 1 ] ) - l e n ( t o k e n s \ [ i ] ) )   f o r   i   i n   r a n g e ( l e n ( t o k e n s ) - 1 ) ]   i f   l e n ( t o k e n s )   >   1   e l s e   \ [ ] 
 
 & n b s p ;               f e a t s   | =   { 
 
 & n b s p ;                       " s e m \ _ 0 " :   f l o a t ( n p . s u m ( d i f f s ) )   i f   d i f f s   e l s e   0 . 0 , 
 
 & n b s p ;                       " s e m \ _ 1 " :   f l o a t ( n p . m e a n ( d i f f s ) )   i f   d i f f s   e l s e   0 . 0 , 
 
 & n b s p ;                       " s e m \ _ 2 " :   f l o a t ( n p . s t d ( d i f f s ) )   i f   d i f f s   e l s e   0 . 0 , 
 
 & n b s p ;                       " s e m \ _ 3 " :   f l o a t ( l e n ( s e t ( \ [ t . l o w e r ( )   f o r   t   i n   t o k e n s ] ) ) ) , 
 
 & n b s p ;                       " s e m \ _ 4 " :   f l o a t ( l e n ( t o k e n s ) ) , 
 
 & n b s p ;                       " s e m \ _ 5 " :   f l o a t ( l e n ( s e t ( w   f o r   w   i n   t o k e n s   i f   l e n ( w ) > 6 ) ) ) , 
 
 & n b s p ;               } 
 
 
 
 & n b s p ;               #   a u x   p a d d i n g   ( z e r o s ) 
 
 & n b s p ;               f o r   i   i n   r a n g e ( 8 ) : 
 
 & n b s p ;                       f e a t s \ [ f " a u x \ _ { i } " ]   =   f e a t s . g e t ( f " a u x \ _ { i } " ,   0 . 0 ) 
 
 
 
 & n b s p ;               #   m i n i m a l   B a n a c h   ( p r a g m a t i c )   f e a t u r e s   f o r   d e m o 
 
 & n b s p ;               f e a t s \ [ " p r a g \ _ l a t e n c y \ _ c l a s s " ]   =   1 . 0 
 
 & n b s p ;               f e a t s \ [ " p r a g \ _ c o s t \ _ c l a s s " ]   =   1 . 0 
 
 & n b s p ;               f e a t s \ [ " p r a g \ _ p i i \ _ l e v e l " ]   =   0 . 0 
 
 & n b s p ;               f e a t s \ [ " p r a g \ _ r e g i o n \ _ e u \ _ o n l y " ]   =   0 . 0 
 
 & n b s p ;               r e t u r n   f e a t s 
 
 ` ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / e n e r g y . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   t y p i n g   i m p o r t   D i c t ,   T u p l e 
 
 f r o m   . m e t r i c   i m p o r t   S y m b o l i c M a n i f o l d M e t r i c 
 
 
 
 c l a s s   S y m b o l i c F r e e E n e r g y : 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ,   a l p h a ,   b e t a \ _ t ,   b e t a \ _ c ,   b e t a \ _ d ,   b e t a \ _ s ) : 
 
 & n b s p ;               s e l f . a l p h a ,   s e l f . b e t a \ _ t ,   s e l f . b e t a \ _ c ,   s e l f . b e t a \ _ d ,   s e l f . b e t a \ _ s   =   a l p h a ,   b e t a \ _ t ,   b e t a \ _ c ,   b e t a \ _ d ,   b e t a \ _ s 
 
 
 
 & n b s p ;       @ p r o p e r t y 
 
 & n b s p ;       d e f   b e t a \ _ d ( s e l f ) :   r e t u r n   s e l f . \ _ b e t a \ _ d 
 
 & n b s p ;       @ b e t a \ _ d . s e t t e r 
 
 & n b s p ;       d e f   b e t a \ _ d ( s e l f ,   v ) :   s e l f . \ _ b e t a \ _ d   =   v 
 
 
 
 & n b s p ;       d e f   c o m p u t e ( s e l f ,   x R :   n p . n d a r r a y ,   m o d e l ,   p r e d i c t o r s :   D i c t ,   c o h e r e n c e ,   m e t r i c :   S y m b o l i c M a n i f o l d M e t r i c 
 
 & n b s p ;                             )   - >   T u p l e \ [ f l o a t ,   f l o a t ,   D i c t \ [ s t r ,   f l o a t ] ] : 
 
 & n b s p ;               d ,   d \ _ s t d   =   m e t r i c . d i s t a n c e ( x R ,   m o d e l . c e n t e r ) 
 
 & n b s p ;               q ,   q \ _ l o ,   q \ _ h i   =   p r e d i c t o r s \ [ " q u a l i t y " ] . p r e d i c t ( \ [ x R ] ) 
 
 & n b s p ;               t ,   t \ _ l o ,   t \ _ h i   =   p r e d i c t o r s \ [ " l a t e n c y " ] . p r e d i c t ( \ [ x R ] ) 
 
 & n b s p ;               c ,   c \ _ l o ,   c \ _ h i   =   p r e d i c t o r s \ [ " c o s t " ] . p r e d i c t ( \ [ x R ] ) 
 
 
 
 & n b s p ;               #   e v i d e n c e   i n   w h i t e n e d   s p a c e 
 
 & n b s p ;               W   =   m e t r i c . W   o r   m e t r i c . \ _ u p d a t e \ _ c h o l e s k y ( ) 
 
 & n b s p ;               x w   =   W   @   ( x R   -   m o d e l . c e n t e r ) 
 
 & n b s p ;               l o g \ _ e   =   c o h e r e n c e . l o g \ _ e v i d e n c e ( m o d e l . n a m e ,   x w ) 
 
 
 
 & n b s p ;               U   =   ( s e l f . a l p h a \ * q \ [ 0 ]   -   s e l f . b e t a \ _ t \ * t \ [ 0 ]   -   s e l f . b e t a \ _ c \ * c \ [ 0 ]   -   s e l f . b e t a \ _ d \ * d   +   s e l f . b e t a \ _ s \ * l o g \ _ e ) 
 
 & n b s p ;               U \ _ v a r   =   ( ( s e l f . a l p h a \ * ( q \ _ h i - q \ _ l o ) / 3 . 9 2 ) \ * \ * 2   +   ( s e l f . b e t a \ _ t \ * ( t \ _ h i - t \ _ l o ) / 3 . 9 2 ) \ * \ * 2   + 
 
 & n b s p ;                                 ( s e l f . b e t a \ _ c \ * ( c \ _ h i - c \ _ l o ) / 3 . 9 2 ) \ * \ * 2   +   ( s e l f . b e t a \ _ d \ * d \ _ s t d ) \ * \ * 2 ) 
 
 & n b s p ;               c o m p s   =   { " q u a l i t y " :   f l o a t ( q \ [ 0 ] ) ,   " l a t e n c y " :   f l o a t ( - t \ [ 0 ] ) ,   " c o s t " :   f l o a t ( - c \ [ 0 ] ) , 
 
 & n b s p ;                                 " d i s t a n c e " :   f l o a t ( - d ) ,   " e v i d e n c e " :   f l o a t ( l o g \ _ e ) ,   " u n c e r t a i n t y " :   f l o a t ( n p . s q r t ( U \ _ v a r ) ) } 
 
 & n b s p ;               r e t u r n   f l o a t ( U ) ,   f l o a t ( n p . s q r t ( U \ _ v a r ) ) ,   c o m p s 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / r o u t e r . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   t i m e ,   j s o n ,   h a s h l i b 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   d a t a c l a s s e s   i m p o r t   d a t a c l a s s 
 
 f r o m   t y p i n g   i m p o r t   D i c t ,   A n y ,   L i s t 
 
 f r o m   . u t i l s   i m p o r t   s p l i t \ _ f e a t u r e s ,   p g d \ _ h a s h 
 
 f r o m   . b o u n d a r y   i m p o r t   B o u n d a r y A n a l y z e r 
 
 f r o m   . c o n s t r a i n t s   i m p o r t   R e f l e c t i v e C o n s t r a i n t S o l v e r 
 
 f r o m   . c o n t r o l   i m p o r t   S R M F C o n t r o l l e r 
 
 f r o m   . e n e r g y   i m p o r t   S y m b o l i c F r e e E n e r g y 
 
 f r o m   . m e t r i c   i m p o r t   S y m b o l i c M a n i f o l d M e t r i c 
 
 
 
 @ d a t a c l a s s 
 
 c l a s s   S w i t c h C e r t i f i c a t e : 
 
 & n b s p ;       m o d e l :   s t r 
 
 & n b s p ;       u t i l i t y :   f l o a t 
 
 & n b s p ;       u t i l i t y \ _ c o m p o n e n t s :   D i c t \ [ s t r ,   f l o a t ] 
 
 & n b s p ;       c o n s t r a i n t s :   D i c t \ [ s t r ,   A n y ] 
 
 & n b s p ;       b o u n d a r y \ _ a n a l y s i s :   D i c t \ [ s t r ,   A n y ] 
 
 & n b s p ;       d r i f t \ _ s t a t u s :   D i c t \ [ s t r ,   f l o a t ] 
 
 & n b s p ;       p g d \ _ s i g n a t u r e :   s t r 
 
 & n b s p ;       t i m e s t a m p :   f l o a t 
 
 & n b s p ;       r o u t e r \ _ v e r s i o n :   s t r   =   " 0 . 1 . 0 " 
 
 
 
 & n b s p ;       d e f   t o \ _ j s o n ( s e l f )   - >   s t r : 
 
 & n b s p ;               r e t u r n   j s o n . d u m p s ( { 
 
 & n b s p ;                       " m o d e l " :   s e l f . m o d e l , 
 
 & n b s p ;                       " u t i l i t y " :   r o u n d ( s e l f . u t i l i t y ,   6 ) , 
 
 & n b s p ;                       " u t i l i t y \ _ c o m p o n e n t s " :   { k :   f l o a t ( v )   f o r   k ,   v   i n   s e l f . u t i l i t y \ _ c o m p o n e n t s . i t e m s ( ) } , 
 
 & n b s p ;                       " c o n s t r a i n t s " :   s e l f . c o n s t r a i n t s , 
 
 & n b s p ;                       " b o u n d a r y " :   s e l f . b o u n d a r y \ _ a n a l y s i s , 
 
 & n b s p ;                       " d r i f t " :   s e l f . d r i f t \ _ s t a t u s , 
 
 & n b s p ;                       " p g d \ _ s i g n a t u r e " :   s e l f . p g d \ _ s i g n a t u r e \ [ : 1 6 ] , 
 
 & n b s p ;                       " t i m e s t a m p " :   s e l f . t i m e s t a m p , 
 
 & n b s p ;                       " r o u t e r \ _ v e r s i o n " :   s e l f . r o u t e r \ _ v e r s i o n 
 
 & n b s p ;               } ,   i n d e n t = 2 ) 
 
 
 
 c l a s s   C o m p i t u m R o u t e r : 
 
 & n b s p ;       d e f   \ _ \ _ i n i t \ _ \ _ ( s e l f ,   m o d e l s :   L i s t \ [ A n y ] ,   p r e d i c t o r s :   D i c t \ [ s t r ,   D i c t ] ,   s o l v e r :   R e f l e c t i v e C o n s t r a i n t S o l v e r , 
 
 & n b s p ;                                 c o h e r e n c e ,   b o u n d a r y :   B o u n d a r y A n a l y z e r ,   s r m f :   S R M F C o n t r o l l e r , 
 
 & n b s p ;                                 p g d \ _ e x t r a c t o r ,   m e t r i c \ _ m a p :   D i c t \ [ s t r ,   S y m b o l i c M a n i f o l d M e t r i c ] , 
 
 & n b s p ;                                 e n e r g y :   S y m b o l i c F r e e E n e r g y ,   u p d a t e \ _ s t r i d e :   i n t   =   8 ) : 
 
 & n b s p ;               s e l f . m o d e l s   =   { m . n a m e :   m   f o r   m   i n   m o d e l s } 
 
 & n b s p ;               s e l f . p r e d i c t o r s   =   p r e d i c t o r s 
 
 & n b s p ;               s e l f . s o l v e r   =   s o l v e r 
 
 & n b s p ;               s e l f . c o h e r e n c e   =   c o h e r e n c e 
 
 & n b s p ;               s e l f . b o u n d a r y   =   b o u n d a r y 
 
 & n b s p ;               s e l f . s r m f   =   s r m f 
 
 & n b s p ;               s e l f . p g d   =   p g d \ _ e x t r a c t o r 
 
 & n b s p ;               s e l f . m e t r i c \ _ m a p   =   m e t r i c \ _ m a p 
 
 & n b s p ;               s e l f . e n e r g y   =   e n e r g y 
 
 & n b s p ;               s e l f . \ _ s t e p   =   0 
 
 & n b s p ;               s e l f . \ _ s t r i d e   =   m a x ( i n t ( u p d a t e \ _ s t r i d e ) ,   1 ) 
 
 
 
 & n b s p ;       d e f   r o u t e ( s e l f ,   p r o m p t :   s t r ,   c o n t e x t :   D i c t \ [ s t r ,   A n y ]   |   N o n e   =   N o n e )   - >   S w i t c h C e r t i f i c a t e : 
 
 & n b s p ;               c o n t e x t   =   c o n t e x t   o r   { } 
 
 & n b s p ;               f e a t s   =   s e l f . p g d . e x t r a c t \ _ f e a t u r e s ( p r o m p t ) 
 
 & n b s p ;               x R \ _ a l l ,   x B   =   s p l i t \ _ f e a t u r e s ( f e a t s ) 
 
 & n b s p ;               u t i l i t i e s ,   c o m p s ,   u \ _ s i g m a s   =   { } ,   { } ,   { } 
 
 
 
 & n b s p ;               f o r   n a m e ,   m o d e l   i n   s e l f . m o d e l s . i t e m s ( ) : 
 
 & n b s p ;                       m e t   =   s e l f . m e t r i c \ _ m a p \ [ n a m e ] 
 
 & n b s p ;                       U ,   s i g ,   u c   =   s e l f . e n e r g y . c o m p u t e ( x R \ _ a l l ,   m o d e l ,   s e l f . p r e d i c t o r s \ [ n a m e ] ,   s e l f . c o h e r e n c e ,   m e t ) 
 
 & n b s p ;                       u t i l i t i e s \ [ s e l f . m o d e l s \ [ n a m e ] ]   =   f l o a t ( U ) 
 
 & n b s p ;                       c o m p s \ [ n a m e ]   =   u c 
 
 & n b s p ;                       u \ _ s i g m a s \ [ n a m e ]   =   f l o a t ( s i g ) 
 
 
 
 & n b s p ;               m \ _ s t a r ,   c i n f o   =   s e l f . s o l v e r . s e l e c t ( x B ,   l i s t ( s e l f . m o d e l s . v a l u e s ( ) ) ,   u t i l i t i e s ) 
 
 & n b s p ;               b i n f o   =   s e l f . b o u n d a r y . a n a l y z e ( { m . n a m e :   u t i l i t i e s \ [ m ]   f o r   m   i n   s e l f . m o d e l s . v a l u e s ( ) } ,   u \ _ s i g m a s ) 
 
 
 
 & n b s p ;               #   A d a p t   m e t r i c   p e r i o d i c a l l y   ( t w o - t i m e s c a l e ) 
 
 & n b s p ;               s e l f . \ _ s t e p   + =   1 
 
 & n b s p ;               g r a d \ _ n o r m   =   1 . 0 
 
 & n b s p ;               i f   s e l f . \ _ s t e p   %   s e l f . \ _ s t r i d e   = =   0 : 
 
 & n b s p ;                       m e t   =   s e l f . m e t r i c \ _ m a p \ [ m \ _ s t a r . n a m e ] 
 
 & n b s p ;                       d \ _ b e s t   =   a b s ( - c o m p s \ [ m \ _ s t a r . n a m e ] \ [ " d i s t a n c e " ] ) 
 
 & n b s p ;                       g r a d \ _ n o r m   =   m e t . u p d a t e \ _ s p d ( x R \ _ a l l ,   s e l f . m o d e l s \ [ m \ _ s t a r . n a m e ] . c e n t e r ,   s e l f . e n e r g y . b e t a \ _ d , 
 
 & n b s p ;                                                                             d \ _ b e s t ,   e t a = 1 e - 2 ,   s r m f \ _ c o n t r o l l e r = s e l f . s r m f ) 
 
 
 
 & n b s p ;               \ _ ,   d r i f t   =   s e l f . s r m f . u p d a t e ( d \ _ s t a r = a b s ( - c o m p s \ [ m \ _ s t a r . n a m e ] \ [ " d i s t a n c e " ] ) ,   g r a d \ _ n o r m = g r a d \ _ n o r m ) 
 
 
 
 & n b s p ;               c e r t   =   S w i t c h C e r t i f i c a t e ( 
 
 & n b s p ;                       m o d e l = m \ _ s t a r . n a m e , 
 
 & n b s p ;                       u t i l i t y = u t i l i t i e s \ [ m \ _ s t a r ] , 
 
 & n b s p ;                       u t i l i t y \ _ c o m p o n e n t s = c o m p s \ [ m \ _ s t a r . n a m e ] , 
 
 & n b s p ;                       c o n s t r a i n t s = c i n f o , 
 
 & n b s p ;                       b o u n d a r y \ _ a n a l y s i s = b i n f o , 
 
 & n b s p ;                       d r i f t \ _ s t a t u s = d r i f t , 
 
 & n b s p ;                       p g d \ _ s i g n a t u r e = p g d \ _ h a s h ( p r o m p t ) , 
 
 & n b s p ;                       t i m e s t a m p = t i m e . t i m e ( ) 
 
 & n b s p ;               ) 
 
 & n b s p ;               r e t u r n   c e r t 
 
 ` ` ` 
 
 
 
 \ # #   ` c o m p i t u m / c l i . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   \ _ \ _ f u t u r e \ _ \ _   i m p o r t   a n n o t a t i o n s 
 
 i m p o r t   j s o n ,   y a m l ,   n u m p y   a s   n p ,   t y p e r 
 
 f r o m   p a t h l i b   i m p o r t   P a t h 
 
 f r o m   t y p i n g   i m p o r t   O p t i o n a l 
 
 f r o m   . p g d   i m p o r t   R e g e x P r o m p t E x t r a c t o r 
 
 f r o m   . m o d e l s   i m p o r t   M o d e l 
 
 f r o m   . c a p a b i l i t i e s   i m p o r t   C a p a b i l i t i e s 
 
 f r o m   . p r e d i c t o r s   i m p o r t   C a l i b r a t e d P r e d i c t o r 
 
 f r o m   . m e t r i c   i m p o r t   S y m b o l i c M a n i f o l d M e t r i c 
 
 f r o m   . c o h e r e n c e   i m p o r t   C o h e r e n c e F u n c t i o n a l 
 
 f r o m   . c o n s t r a i n t s   i m p o r t   R e f l e c t i v e C o n s t r a i n t S o l v e r 
 
 f r o m   . b o u n d a r y   i m p o r t   B o u n d a r y A n a l y z e r 
 
 f r o m   . c o n t r o l   i m p o r t   S R M F C o n t r o l l e r 
 
 f r o m   . e n e r g y   i m p o r t   S y m b o l i c F r e e E n e r g y 
 
 f r o m   . r o u t e r   i m p o r t   C o m p i t u m R o u t e r 
 
 
 
 a p p   =   t y p e r . T y p e r ( h e l p = " c o m p i t u m   C L I " ) 
 
 
 
 d e f   \ _ l o a d \ _ c o n s t r a i n t s ( p a t h :   P a t h ) : 
 
 & n b s p ;       c f g   =   y a m l . s a f e \ _ l o a d ( p a t h . r e a d \ _ t e x t ( ) ) 
 
 & n b s p ;       i m p o r t   n u m p y   a s   n p 
 
 & n b s p ;       r e t u r n   n p . a r r a y ( c f g \ [ " A " ] ,   f l o a t ) ,   n p . a r r a y ( c f g \ [ " b " ] ,   f l o a t ) 
 
 
 
 d e f   \ _ t o y \ _ m o d e l s ( D :   i n t ) : 
 
 & n b s p ;       r n g   =   n p . r a n d o m . d e f a u l t \ _ r n g ( 7 ) 
 
 & n b s p ;       c e n t e r s   =   { 
 
 & n b s p ;               " f a s t " :         r n g . n o r m a l ( 0 . 0 ,   0 . 4 ,   s i z e = D ) , 
 
 & n b s p ;               " t h i n k i n g " : r n g . n o r m a l ( 0 . 0 ,   1 . 0 ,   s i z e = D ) , 
 
 & n b s p ;               " a u t o " :         r n g . n o r m a l ( 0 . 1 ,   0 . 7 ,   s i z e = D ) 
 
 & n b s p ;       } 
 
 & n b s p ;       c a p s   =   C a p a b i l i t i e s ( r e g i o n s = { " U S " , " C A " , " E U " } ,   t o o l s \ _ a l l o w e d = { " n o n e " } ) 
 
 & n b s p ;       r e t u r n   \ [ M o d e l ( n a m e = k ,   c e n t e r = v ,   c a p a b i l i t i e s = c a p s )   f o r   k ,   v   i n   c e n t e r s . i t e m s ( ) ] 
 
 
 
 @ a p p . c o m m a n d ( ) 
 
 d e f   r o u t e ( p r o m p t :   s t r , 
 
 & n b s p ;                   c o n s t r a i n t s :   P a t h   =   P a t h ( " c o n f i g s / c o n s t r a i n t s \ _ u s \ _ d e f a u l t . y a m l " ) , 
 
 & n b s p ;                   d e f a u l t s :   P a t h   =   P a t h ( " c o n f i g s / r o u t e r \ _ d e f a u l t s . y a m l " ) , 
 
 & n b s p ;                   v e r b o s e :   b o o l   =   F a l s e ) : 
 
 & n b s p ;       d c f g   =   y a m l . s a f e \ _ l o a d ( d e f a u l t s . r e a d \ _ t e x t ( ) ) 
 
 & n b s p ;       D   =   i n t ( d c f g \ [ " m e t r i c " ] \ [ " D " ] ) 
 
 & n b s p ;       r a n k   =   i n t ( d c f g \ [ " m e t r i c " ] \ [ " r a n k " ] ) 
 
 & n b s p ;       d e l t a   =   f l o a t ( d c f g \ [ " m e t r i c " ] \ [ " d e l t a " ] ) 
 
 
 
 & n b s p ;       m o d e l s   =   \ _ t o y \ _ m o d e l s ( D ) 
 
 & n b s p ;       p r e d i c t o r s   =   { 
 
 & n b s p ;               m . n a m e :   { " q u a l i t y " :   C a l i b r a t e d P r e d i c t o r ( ) ,   " l a t e n c y " :   C a l i b r a t e d P r e d i c t o r ( ) ,   " c o s t " :   C a l i b r a t e d P r e d i c t o r ( ) } 
 
 & n b s p ;               f o r   m   i n   m o d e l s 
 
 & n b s p ;       } 
 
 & n b s p ;       #   q u i c k   s y n t h e t i c   f i t   f o r   d e m o 
 
 & n b s p ;       X \ _ d e m o   =   n p . r a n d o m . r a n d n ( 5 1 2 ,   D ) 
 
 & n b s p ;       f o r   m   i n   m o d e l s : 
 
 & n b s p ;               y q   =   0 . 6   +   0 . 1 \ * n p . t a n h ( X \ _ d e m o   @   ( m . c e n t e r / n p . l i n a l g . n o r m ( m . c e n t e r ) + 1 e - 8 ) ) 
 
 & n b s p ;               y t   =   0 . 5   +   0 . 5 \ * n p . a b s ( X \ _ d e m o   @   n p . o n e s ( D ) / n p . s q r t ( D ) ) 
 
 & n b s p ;               y c   =   0 . 2   +   0 . 4 \ * n p . a b s ( X \ _ d e m o   @   ( n p . a r a n g e ( D ) / D ) ) 
 
 & n b s p ;               p r e d i c t o r s \ [ m . n a m e ] \ [ " q u a l i t y " ] . f i t ( X \ _ d e m o ,   y q ) 
 
 & n b s p ;               p r e d i c t o r s \ [ m . n a m e ] \ [ " l a t e n c y " ] . f i t ( X \ _ d e m o ,   y t ) 
 
 & n b s p ;               p r e d i c t o r s \ [ m . n a m e ] \ [ " c o s t " ] . f i t ( X \ _ d e m o ,   y c ) 
 
 
 
 & n b s p ;       m e t r i c s   =   { m . n a m e :   S y m b o l i c M a n i f o l d M e t r i c ( D ,   r a n k ,   d e l t a )   f o r   m   i n   m o d e l s } 
 
 & n b s p ;       c o h e r e n c e   =   C o h e r e n c e F u n c t i o n a l ( k = 5 0 0 ) 
 
 & n b s p ;       A , b   =   \ _ l o a d \ _ c o n s t r a i n t s ( c o n s t r a i n t s ) 
 
 & n b s p ;       s o l v e r   =   R e f l e c t i v e C o n s t r a i n t S o l v e r ( A ,   b ) 
 
 & n b s p ;       b o u n d a r y   =   B o u n d a r y A n a l y z e r ( ) 
 
 & n b s p ;       s r m f   =   S R M F C o n t r o l l e r ( ) 
 
 & n b s p ;       e n e r g y   =   S y m b o l i c F r e e E n e r g y ( d c f g \ [ " a l p h a " ] ,   d c f g \ [ " b e t a \ _ t " ] ,   d c f g \ [ " b e t a \ _ c " ] ,   d c f g \ [ " b e t a \ _ d " ] ,   d c f g \ [ " b e t a \ _ s " ] ) 
 
 & n b s p ;       p g d   =   R e g e x P r o m p t E x t r a c t o r ( ) 
 
 
 
 & n b s p ;       r o u t e r   =   C o m p i t u m R o u t e r ( m o d e l s ,   p r e d i c t o r s ,   s o l v e r ,   c o h e r e n c e ,   b o u n d a r y ,   s r m f ,   p g d ,   m e t r i c s ,   e n e r g y ,   u p d a t e \ _ s t r i d e = i n t ( d c f g \ [ " u p d a t e \ _ s t r i d e " ] ) ) 
 
 & n b s p ;       c e r t   =   r o u t e r . r o u t e ( p r o m p t ) 
 
 & n b s p ;       t y p e r . e c h o ( c e r t . t o \ _ j s o n ( )   i f   v e r b o s e   e l s e   j s o n . d u m p s ( { " m o d e l " :   c e r t . m o d e l ,   " U " :   c e r t . u t i l i t y } ,   i n d e n t = 2 ) ) 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 \ #   5 )   T e s t s 
 
 
 
 \ # #   ` t e s t s / t e s t \ _ i n v a r i a n t s . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   c o m p i t u m . m e t r i c   i m p o r t   S y m b o l i c M a n i f o l d M e t r i c 
 
 
 
 d e f   t e s t \ _ s p d \ _ p r o p e r t i e s ( ) : 
 
 & n b s p ;       m   =   S y m b o l i c M a n i f o l d M e t r i c ( 2 0 ,   5 ) 
 
 & n b s p ;       M   =   m . m e t r i c \ _ m a t r i x ( ) 
 
 & n b s p ;       a s s e r t   n p . a l l c l o s e ( M ,   M . T ) 
 
 & n b s p ;       e i g   =   n p . l i n a l g . e i g v a l s h ( M ) 
 
 & n b s p ;       a s s e r t   n p . a l l ( e i g   >   0 ) 
 
 
 
 d e f   t e s t \ _ t r i a n g l e \ _ i n e q u a l i t y ( ) : 
 
 & n b s p ;       m   =   S y m b o l i c M a n i f o l d M e t r i c ( 1 2 ,   4 ) 
 
 & n b s p ;       x ,   y ,   z   =   n p . r a n d o m . r a n d n ( 1 2 ) ,   n p . r a n d o m . r a n d n ( 1 2 ) ,   n p . r a n d o m . r a n d n ( 1 2 ) 
 
 & n b s p ;       d \ _ x y ,   \ _   =   m . d i s t a n c e ( x ,   y ) 
 
 & n b s p ;       d \ _ y z ,   \ _   =   m . d i s t a n c e ( y ,   z ) 
 
 & n b s p ;       d \ _ x z ,   \ _   =   m . d i s t a n c e ( x ,   z ) 
 
 & n b s p ;       a s s e r t   d \ _ x z   < =   d \ _ x y   +   d \ _ y z   +   1 e - 9 
 
 
 
 d e f   t e s t \ _ w h i t e n i n g \ _ i s o m e t r y ( ) : 
 
 & n b s p ;       m   =   S y m b o l i c M a n i f o l d M e t r i c ( 1 0 ,   3 ) ;   m . \ _ u p d a t e \ _ c h o l e s k y ( ) 
 
 & n b s p ;       a ,   b   =   n p . r a n d o m . r a n d n ( 1 0 ) ,   n p . r a n d o m . r a n d n ( 1 0 ) 
 
 & n b s p ;       d ,   \ _   =   m . d i s t a n c e ( a ,   b ) 
 
 & n b s p ;       w a ,   w b   =   m . W   @   a ,   m . W   @   b 
 
 & n b s p ;       a s s e r t   n p . i s c l o s e ( d ,   n p . l i n a l g . n o r m ( w a   -   w b ) ,   r t o l = 1 e - 9 ) 
 
 ` ` ` 
 
 
 
 \ # #   ` t e s t s / t e s t \ _ b o u n d a r y . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   c o m p i t u m . b o u n d a r y   i m p o r t   B o u n d a r y A n a l y z e r 
 
 
 
 d e f   t e s t \ _ b o u n d a r y \ _ l o g i c ( ) : 
 
 & n b s p ;       b   =   B o u n d a r y A n a l y z e r ( ) 
 
 & n b s p ;       u t i l i t i e s   =   { " f a s t " :   0 . 5 0 ,   " t h i n k i n g " :   0 . 5 2 ,   " a u t o " :   0 . 4 8 } 
 
 & n b s p ;       u \ _ s i g m a   =   { " f a s t " :   0 . 0 5 ,   " t h i n k i n g " :   0 . 2 ,   " a u t o " :   0 . 0 5 } 
 
 & n b s p ;       i n f o   =   b . a n a l y z e ( u t i l i t i e s ,   u \ _ s i g m a ) 
 
 & n b s p ;       a s s e r t   " i s \ _ b o u n d a r y "   i n   i n f o 
 
 ` ` ` 
 
 
 
 \ # #   ` t e s t s / t e s t \ _ c o n s t r a i n t s . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   d a t a c l a s s e s   i m p o r t   d a t a c l a s s 
 
 f r o m   c o m p i t u m . c o n s t r a i n t s   i m p o r t   R e f l e c t i v e C o n s t r a i n t S o l v e r 
 
 f r o m   c o m p i t u m . c a p a b i l i t i e s   i m p o r t   C a p a b i l i t i e s 
 
 
 
 @ d a t a c l a s s 
 
 c l a s s   M : 
 
 & n b s p ;       n a m e :   s t r 
 
 & n b s p ;       c a p a b i l i t i e s :   C a p a b i l i t i e s 
 
 
 
 d e f   t e s t \ _ s o l v e r \ _ b a s i c ( ) : 
 
 & n b s p ;       A   =   n p . e y e ( 2 ) ;   b   =   n p . a r r a y ( \ [ 1 . 0 ,   1 . 0 ] ) 
 
 & n b s p ;       s o l v e r   =   R e f l e c t i v e C o n s t r a i n t S o l v e r ( A ,   b ) 
 
 & n b s p ;       p g d   =   n p . a r r a y ( \ [ 0 . 5 ,   0 . 0 ] ) 
 
 & n b s p ;       m o d e l s   =   \ [ M ( " a " ,   C a p a b i l i t i e s ( s e t ( ) ,   s e t ( ) ) ) ,   M ( " b " ,   C a p a b i l i t i e s ( s e t ( ) ,   s e t ( ) ) ) ] 
 
 & n b s p ;       u t i l i t i e s   =   { m o d e l s \ [ 0 ] :   0 . 2 ,   m o d e l s \ [ 1 ] :   0 . 3 } 
 
 & n b s p ;       m \ _ s t a r ,   i n f o   =   s o l v e r . s e l e c t ( p g d ,   m o d e l s ,   u t i l i t i e s ) 
 
 & n b s p ;       a s s e r t   m \ _ s t a r . n a m e   = =   " b " 
 
 & n b s p ;       a s s e r t   i n f o \ [ " f e a s i b l e " ]   i s   T r u e 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 \ #   6 )   E x a m p l e s 
 
 
 
 \ # #   ` e x a m p l e s / s y n t h \ _ b e n c h . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 i m p o r t   n u m p y   a s   n p 
 
 f r o m   c o m p i t u m . m e t r i c   i m p o r t   S y m b o l i c M a n i f o l d M e t r i c 
 
 
 
 d e f   m a i n ( ) : 
 
 & n b s p ;       r n g   =   n p . r a n d o m . d e f a u l t \ _ r n g ( 0 ) 
 
 & n b s p ;       D   =   3 5 
 
 & n b s p ;       M   =   S y m b o l i c M a n i f o l d M e t r i c ( D ,   8 ) 
 
 & n b s p ;       #   t w o   c l u s t e r s :   m a t h - l i k e   v s   c o d e - l i k e 
 
 & n b s p ;       m a t h \ _ c e n t e r   =   r n g . n o r m a l ( 0 ,   1 ,   s i z e = D ) 
 
 & n b s p ;       c o d e \ _ c e n t e r   =   r n g . n o r m a l ( 0 ,   1 ,   s i z e = D ) ;   c o d e \ _ c e n t e r \ [ : 5 ]   + =   2 . 0 
 
 & n b s p ;       X \ _ m a t h   =   r n g . n o r m a l ( 0 ,   0 . 6 ,   s i z e = ( 5 0 0 ,   D ) )   +   m a t h \ _ c e n t e r 
 
 & n b s p ;       X \ _ c o d e   =   r n g . n o r m a l ( 0 ,   0 . 6 ,   s i z e = ( 5 0 0 ,   D ) )   +   c o d e \ _ c e n t e r 
 
 & n b s p ;       d m   =   n p . m e a n ( \ [ M . d i s t a n c e ( x ,   m a t h \ _ c e n t e r ) \ [ 0 ]   f o r   x   i n   X \ _ m a t h ] ) 
 
 & n b s p ;       d c   =   n p . m e a n ( \ [ M . d i s t a n c e ( x ,   c o d e \ _ c e n t e r ) \ [ 0 ]   f o r   x   i n   X \ _ c o d e ] ) 
 
 & n b s p ;       p r i n t ( { " a v g \ _ d \ _ m a t h " :   f l o a t ( d m ) ,   " a v g \ _ d \ _ c o d e " :   f l o a t ( d c ) } ) 
 
 i f   \ _ \ _ n a m e \ _ \ _   = =   " \ _ \ _ m a i n \ _ \ _ " : 
 
 & n b s p ;       m a i n ( ) 
 
 ` ` ` 
 
 
 
 \ # #   ` e x a m p l e s / d e m o \ _ r o u t e . p y ` 
 
 
 
 ` ` ` p y t h o n 
 
 f r o m   s u b p r o c e s s   i m p o r t   r u n 
 
 r u n ( \ [ " c o m p i t u m " , " r o u t e " , " - - p r o m p t " , " P r o v e   t h a t   t h e   h a r m o n i c   s e r i e s   d i v e r g e s . " ] ,   c h e c k = T r u e ) 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 \ # #   T h a t â € ™ s   i t 
 
 
 
 \ *   R u n   w i t h   \ * \ * G e m i n i \ * \ * :   ` g e m i n i   r u n   s e t u p   \ & \ &   g e m i n i   r u n   t e s t   \ & \ &   g e m i n i   r u n   r o u t e - d e m o ` 
 
 \ *   O r   p l a i n   s h e l l :   ` m a k e   s e t u p   t e s t   d e m o ` 
 
 
 
 T h i s   p a c k a g e   g i v e s   y o u r   t e a m   a   \ * \ * c o n t a i n e d ,   p r o d u c t i o n - r e a d y   s t a r t i n g   p o i n t \ * \ * :   h a r d e n e d   g e o m e t r y ,   c o n s t r a i n t   s o l v e r   w /   s h a d o w   p r i c e s ,   w h i t e n e d   K D E ,   S R M F   t r u s t   r e g i o n ,   C L I ,   t e s t s ,   a n d   a   b e n c h .   H o o k   y o u r   r e a l   p r e d i c t o r s ,   r e a l   P G D   e x t r a c t o r s ,   a n d   y o u r   P y L a n t e r n   o b s e r v e r s   i n t o   t h e   s e a m s   a l r e a d y   e x p o s e d . 
 
 
 
