File:
data.txt includes 98209 rounds of Chinese Standard Mahjong game playing records. The file is large, so it is recommended not to open the file in text editors.
sample.txt includes the first 16 game playing records, for the purpose of viewing and checking the record formats.

Data format used for each round:
(1) Match <ID> on the first line, which can be viewed by opening the link https://botzone.org.cn/match/<ID> in web browsers. 
(2) Wind 0..3 on the second line, representing the round wind (0~3 represent east, north, west, and south).
(3) Player <N> Deal XX XX ...for the next four lines, representing the initial 13 tiles for four players. Note Player 0~3 representing the East, North, West, and South player specifically. All tiles can be represented using "capitalized letter+number", i.e. "W4" represents "Character-4", "B6" represents "Dot-6", "T8" represents "Bamboo-8", "F1"~"F4" represent "East, South, West, North Wind, "J1"~"J3" represent "Red Dragon, Green Dragon, White Dragon".
(4) The next game playing rounds will have the following formats:
Player <N> Draw XX # player draws a tile, XX represents the tile
Player <N> Play XX # player discards a tile, XX represents the tile
Player <N> Chi XX # player chows a tile (to form a Sequence), XX represents the middle tile of the sequence. I.e. If the previous player discarded B7, the formed sequence is B7B8B9, then XX will be B8.
Player <N> Peng XX # player pongs a tile (to form a Triplet), XX represents the tile got ponged, which is also the tile the previous player discarded. 
Player <N> Gang XX # player kongs a tile (to form an exposed Kong), XX represents the tile got ponged, which is also the tile the previous player discarded. 
Player <N> AnGang XX # player kongs a tile (to form a concealed Kong), XX represents the tile that forms a kong, not necessarily equal to the tile that the player draws. 
Player <N> BuGang XX # player adds a tile to a melded pung (to form a exposed Kong), XX represents the tile.
Player <N> Hu XX # player wins the round. XX represents the last tile to complete the hand, could be a self-drawn tile, a tile discarded by other players, or robbing a kong.


[NOTE] For Peng, Gang, Hu, format, it is possible that one or more "Ignore Player <N> Chi/Peng/Gang/Hu XX", representing after previous player's turn, multiple players can declare Chi/Peng/Gang/Hu, (priority: Hu>Peng=Gang>Chi, under same priority, players take counter-closewise turn), Ignore indicate an operation that is ignored.

Example:
Player 2 Play B3
Player 1 Hu B3 Ignore Player 0 PENG B3 Ignore Player 3 CHI B4
Meaning Player 2 discards B3, Palyer 3 can eat, Player 0 can Peng, Player 1 can Hu, in actual game, Player 1 Hu, other two operations are ignored. 

Player 1 Play W7
Player 2 Hu W7 Ignore Player 0 HU W7
Meaning Player 1 plays W7, Player 0 and Player 2 can Hu, but Player 2 takes the precedence, Player 0's play is ignored.

(5) A round ends under two situations, either a player Hu, or the game plays til the end but no one wins. 
For cases in which a player Hu, the ending info shows the following in the last two lines:
Fan <F> <Description> (Fan points and Fan breakdown details)
Score <S1> <S2> <S3> <S4> (scores after the game)

For cases in which no one wins, the ending info shows the following in the last two lines:
Huang
Score 0 0 0 0